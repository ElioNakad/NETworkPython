import mysql.connector
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# DATABASE CONNECTION
# ==========================================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="fypdb"
)

cursor = conn.cursor(dictionary=True)

# ==========================================
# CLEANING FUNCTION (VERY IMPORTANT)
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

    # Remove phone numbers
    text = re.sub(r'\+?\d+', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    words = text.split()

    # Remove structural words
    words = [w for w in words if w not in STRUCTURAL_WORDS]

    return " ".join(words).strip()


# ==========================================
# BUILD PROFILE TEXT (FORMATTED VERSION)
# ==========================================

def build_profile_text(user_id):

    # CONTACT
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

    if cv_row and cv_row["cv"]:
        cv_text = cv_row["cv"]
    else:
        cv_text = "None"

    # REVIEWS WITH CONTEXT
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
# FETCH USERS
# ==========================================

cursor.execute("SELECT user_id FROM user_profile_embeddings")
rows = cursor.fetchall()

display_texts = []
cleaned_texts = []
user_ids = []

for row in rows:
    uid = row["user_id"]
    formatted_text = build_profile_text(uid)

    if formatted_text:
        cleaned_text = clean_profile_text(formatted_text)

        display_texts.append(formatted_text)
        cleaned_texts.append(cleaned_text)
        user_ids.append(uid)

# ==========================================
# TRAIN TF-IDF (REAL TRAINING HAPPENS HERE)
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    min_df=1,
    max_df=0.8
)

X = vectorizer.fit_transform(cleaned_texts)

# ==========================================
# STORE VECTORS
# ==========================================

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

print("✅ Profiles rebuilt, cleaned, and vectorized successfully.")

cursor.close()
conn.close()