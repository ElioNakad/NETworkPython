import json
import numpy as np
from app.core.database import get_cursor
from app.utils.math import cosine

def retrieve_candidates(user_id: int, query_embedding, top_k: int = 20):
    cur = get_cursor()

    cur.execute("""
        SELECT uce.contact_id, uce.embedding, uce.profile_text,
               uc.display_name, c.phone
        FROM user_contact_embeddings uce
        JOIN contacts c ON c.id = uce.contact_id
        LEFT JOIN user_contacts uc
          ON uc.user_id = uce.user_id
         AND uc.contact_id = uce.contact_id
        WHERE uce.user_id = %s
    """, (user_id,))

    results = []
    for r in cur.fetchall():
        emb = np.array(json.loads(r["embedding"]), dtype=np.float32)
        score = cosine(query_embedding, emb)

        results.append({
            "contact_id": r["contact_id"],
            "name": r["display_name"] or "(no name)",
            "phone": r["phone"],
            "profile_text": r["profile_text"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results[:top_k]):
        r["idx"] = i

    return results[:top_k]
