import os
import json
import hashlib
import mysql.connector
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# ENV SETUP
# =========================
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)

cursor = db.cursor(dictionary=True)

# =========================
# HELPERS
# =========================
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


def fetch_all(query: str, params: tuple):
    cursor.execute(query, params)
    return cursor.fetchall()


def fetch_one(query: str, params: tuple):
    cursor.execute(query, params)
    return cursor.fetchone()


# =========================
# MAIN QUERY
# =========================
TEST_EMBEDDING_IDS = (604, 601, 109, 13, 601, 148,192,1162,1155,36)

cursor.execute(
    """
    SELECT
        uce.id AS embedding_id,
        uce.user_id,
        uce.contact_id,
        c.phone,
        uc.id AS user_contact_id,
        uc.display_name,
        cu.id AS contact_user_id,
        cu.fname,
        cu.lname
    FROM user_contact_embeddings uce
    JOIN contacts c ON c.id = uce.contact_id
    LEFT JOIN user_contacts uc
           ON uc.user_id = uce.user_id
          AND uc.contact_id = uce.contact_id
    LEFT JOIN users cu
           ON cu.phone = c.phone
    WHERE uce.id IN (%s, %s, %s, %s, %s, %s,%s,%s,%s,%s)
      AND uce.needs_rebuild = 1
    """,
    TEST_EMBEDDING_IDS
)

rows = cursor.fetchall()
print(f"🔄 {len(rows)} embeddings to rebuild")

# =========================
# PROCESS
# =========================
for r in rows:

    # -------- NAME RESOLUTION --------
    if r["display_name"]:
        name = r["display_name"]
    elif r["fname"] or r["lname"]:
        name = f"{r['fname'] or ''} {r['lname'] or ''}".strip()
    else:
        name = "Unknown"

    # -------- DEFAULT IDENTITY --------
    default_identity = "None"
    if r["contact_user_id"]:
        rows_dd = fetch_all(
            """
            SELECT label, description
            FROM default_description
            WHERE users_id = %s
            """,
            (r["contact_user_id"],)
        )

        if rows_dd:
            default_identity = "\n".join(
                f"- {d['label']}: {d['description']}"
                for d in rows_dd
            )

    # -------- PERSONAL LABELS --------
    personal_labels = "None yet"
    if r["user_contact_id"]:
        rows_pl = fetch_all(
            """
            SELECT label, description
            FROM user_contact_descriptions
            WHERE user_contact_id = %s
            """,
            (r["user_contact_id"],)
        )

        if rows_pl:
            personal_labels = "\n".join(
                f"- {p['label']}: {p['description']}"
                for p in rows_pl
            )

    # -------- CV --------
    cv_text = "None"
    if r["contact_user_id"]:
        cv = fetch_one(
            """
            SELECT cv
            FROM users_cv
            WHERE user_id = %s
            """,
            (r["contact_user_id"],)
        )

        if cv and cv["cv"]:
            cv_text = cv["cv"]

    # -------- REVIEWS (WITH DESCRIPTION CONTEXT) --------
    reviews_text = "None"
    if r["contact_user_id"]:
        reviews = fetch_all(
            """
            SELECT
                r.review,
                dd.label AS desc_label,
                dd.description AS desc_description
            FROM reviews r
            JOIN default_description dd
                ON dd.id = r.default_description_id
            WHERE dd.users_id = %s
            """,
            (r["contact_user_id"],)
        )

        if reviews:
            formatted_reviews = []
            for rv in reviews:
                if rv["review"]:
                    formatted_reviews.append(
                        f"""[REVIEW CONTEXT]
                        Role: {rv['desc_label'] or 'Unknown'}
                        Description: {rv['desc_description'] or 'None'}
                        Review: {rv['review']}
                    """
                    )

            if formatted_reviews:
                reviews_text = "\n".join(formatted_reviews)

    # -------- PROFILE TEXT --------
    profile_text = f"""CONTACT
Name: {name}
Phone: {r['phone']}

DEFAULT IDENTITY
{default_identity}

MY PERSONAL LABELS
{personal_labels}

CV
{cv_text}

REVIEWS
{reviews_text}
"""

    context_hash = sha256(profile_text)
    embedding = embed(profile_text)

    cursor.execute(
        """
        UPDATE user_contact_embeddings
        SET profile_text = %s,
            embedding = %s,
            context_hash = %s,
            needs_rebuild = 0
        WHERE id = %s
        """,
        (
            profile_text,
            json.dumps(embedding),
            context_hash,
            r["embedding_id"]
        )
    )

    db.commit()
    print(f"✅ Updated embedding {r['embedding_id']} ({name})")

# =========================
# CLEANUP
# =========================
cursor.close()
db.close()
print("🎉 All embeddings rebuilt correctly")
