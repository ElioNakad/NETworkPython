import mysql.connector
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from app.db import get_db  # IMPORTANT (use your FastAPI DB helper)


# ==========================================
# CLEANING FUNCTION
# ==========================================

STRUCTURAL_WORDS = {
    "contact", "name", "phone",
    "default", "identity",
    "cv", "reviews",
    "review", "role", "description",
    "none"
}

def clean_profile_text(text: str):
    text = text.lower()

    text = re.sub(r'\+?\d+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in STRUCTURAL_WORDS]

    return " ".join(words).strip()


# ==========================================
# BUILD PROFILE TEXT
# ==========================================

def build_profile_text(cursor, user_id):

    cursor.execute(
        "SELECT fname, lname, phone FROM users WHERE id = %s",
        (user_id,)
    )
    user = cursor.fetchone()

    if not user:
        return None

    name = f"{user['fname']} {user['lname']}".strip()
    phone = user['phone'] if user['phone'] else "None"

    # DEFAULT IDENTITY
    cursor.execute(
        "SELECT id, label, description FROM default_description WHERE users_id = %s",
        (user_id,)
    )
    defaults = cursor.fetchall()

    if defaults:
        default_text = "\n".join(
            f"{row['label']}: {row['description']}"
            for row in defaults
            if row['label'] or row['description']
        )
    else:
        default_text = "None"

    # CV
    cursor.execute(
        "SELECT cv FROM users_cv WHERE user_id = %s ORDER BY id DESC LIMIT 1",
        (user_id,)
    )
    cv_row = cursor.fetchone()

    cv_text = cv_row["cv"] if cv_row and cv_row["cv"] else "None"

    # REVIEWS
    cursor.execute("""
        SELECT r.review,
               d.label,
               d.description
        FROM reviews r
        JOIN default_description d
            ON r.default_description_id = d.id
        WHERE d.users_id = %s
    """, (user_id,))

    review_rows = cursor.fetchall()

    if review_rows:
        review_blocks = []

        for row in review_rows:
            block = f"""
[REVIEW CONTEXT]
Role: {row['label']}
Description: {row['description']}

Review:
{row['review']}
""".strip()

            review_blocks.append(block)

        review_text = "\n\n".join(review_blocks)
    else:
        review_text = "None"

    profile_text = f"""
CONTACT:
Name: {name}
Phone: {phone}

DEFAULT IDENTITY:
{default_text}

CV:
{cv_text}

REVIEWS:
{review_text}
""".strip()

    return profile_text


# ==========================================
# MAIN REBUILD FUNCTION
# ==========================================

def rebuild_all_vectors():

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT user_id FROM user_profile_embeddings")
    rows = cursor.fetchall()

    display_texts = []
    cleaned_texts = []
    user_ids = []

    for row in rows:
        uid = row["user_id"]
        formatted_text = build_profile_text(cursor, uid)

        if formatted_text:
            cleaned_text = clean_profile_text(formatted_text)

            display_texts.append(formatted_text)
            cleaned_texts.append(cleaned_text)
            user_ids.append(uid)

    if not cleaned_texts:
        return {"message": "No profiles found"}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1,
        max_df=0.8
    )

    X = vectorizer.fit_transform(cleaned_texts)

    for i, user_id in enumerate(user_ids):
        vector = X[i].toarray()[0]
        vector_json = json.dumps(vector.tolist())

        cursor.execute("""
            UPDATE user_profile_embeddings
            SET profile_text = %s,
                vector_data = %s,
                needs_rebuild = 0
            WHERE user_id = %s
        """, (display_texts[i], vector_json, user_id))

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "message": "Profiles rebuilt successfully",
        "profiles_processed": len(user_ids)
    }