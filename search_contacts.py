import os, json
import numpy as np
import mysql.connector
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

USER_ID = 27
TOP_K = 1

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)
cur = db.cursor(dictionary=True)

def get_query_embedding(text: str) -> np.ndarray:
    # Embeddings API reference: input string -> vector :contentReference[oaicite:3]{index=3}
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def load_user_embeddings(user_id: int):
    # Join to get phone + display_name (since contacts has only phone)
    cur.execute(
        """
        SELECT
            uce.id AS embedding_row_id,
            uce.contact_id,
            c.phone,
            uc.display_name,
            uce.embedding,
            uce.profile_text
        FROM user_contact_embeddings uce
        JOIN contacts c ON c.id = uce.contact_id
        LEFT JOIN user_contacts uc
               ON uc.user_id = uce.user_id
              AND uc.contact_id = uce.contact_id
        WHERE uce.user_id = %s
        """,
        (user_id,)
    )
    return cur.fetchall()

def main():
    query = input("Search prompt (e.g., 'best AI developer'): ").strip()
    if not query:
        print("Empty query, exiting.")
        return

    q = get_query_embedding(query)
    rows = load_user_embeddings(USER_ID)

    scored = []
    for r in rows:
        try:
            emb = np.array(json.loads(r["embedding"]), dtype=np.float32)
        except Exception:
            continue

        score = cosine(q, emb)

        name = r["display_name"] or "Unknown"
        scored.append({
            "score": score,
            "embedding_row_id": r["embedding_row_id"],
            "contact_id": r["contact_id"],
            "name": name,
            "phone": r["phone"],
            "profile_text": (r["profile_text"] or "")[:400]  # preview
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    print("\n=== TOP MATCHES ===")
    for i, s in enumerate(scored[:TOP_K], start=1):
        print(f"\n#{i}  score={s['score']:.4f}")
        print(f"   name: {s['name']}")
        print(f"   phone: {s['phone']}")
        print(f"   embedding_row_id: {s['embedding_row_id']}, contact_id: {s['contact_id']}")
        print("   profile preview:")
        print("   " + s["profile_text"].replace("\n", "\n   "))

if __name__ == "__main__":
    try:
        main()
    finally:
        cur.close()
        db.close()
