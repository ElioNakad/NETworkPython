from app.services.embedding_service import get_embedding
from app.db import get_db
import numpy as np
import json

# This is NOT a hard semantic truth threshold
# It is a "reasonable relevance" cutoff for referral
MIN_REFERRAL_SCORE = 0.20


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def referral_search(my_user_id: int, prompt: str):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # 1️⃣ Get my contacts who are real users AND allow referrals
    cursor.execute("""
        SELECT
            uc.display_name,
            u.id   AS referrer_user_id,
            u.phone
        FROM user_contacts uc
        JOIN contacts c ON c.id = uc.contact_id
        JOIN users u ON u.phone = c.phone
        WHERE uc.user_id = %s
          AND u.refer = 'true'
    """, (my_user_id,))

    referrers = cursor.fetchall()
    if not referrers:
        return []

    # 2️⃣ Embed the prompt ONCE
    prompt_embedding = get_embedding(prompt)

    results = []

    # 3️⃣ For each referrer X → search X’s embeddings
    for x in referrers:
        cursor.execute("""
            SELECT embedding
            FROM user_contact_embeddings
            WHERE user_id = %s
              AND needs_rebuild = 0
        """, (x["referrer_user_id"],))

        rows = cursor.fetchall()
        if not rows:
            continue

        # 4️⃣ Rank embeddings instead of hard thresholding
        best_score = 0.0

        for row in rows:
            embedding = np.array(json.loads(row["embedding"]))
            score = cosine_similarity(prompt_embedding, embedding)
            best_score = max(best_score, score)

        # 5️⃣ Accept referrer if ANY of their contacts is relevant
        if best_score >= MIN_REFERRAL_SCORE:
            results.append({
                "name": x["display_name"],
                "phone": x["phone"],
                "confidence": round(best_score, 2)
            })

    # 6️⃣ Sort referrers by confidence (best first)
    results.sort(key=lambda r: r["confidence"], reverse=True)

    return results
