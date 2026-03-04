import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.db import get_db


def get_recommendations_for_user(user_id: int, top_n: int = 5):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # ---------------------------------------
    # 1) Fetch embeddings excluding:
    # - self
    # - already in contacts
    # ---------------------------------------
    cursor.execute("""
        SELECT u.id, u.fname, u.lname, u.phone, e.vector_data
        FROM user_profile_embeddings e
        JOIN users u ON u.id = e.user_id
        WHERE 
            e.vector_data IS NOT NULL
            AND u.refer = 'true'
            AND u.id != %s
            AND u.phone IS NOT NULL
            AND u.phone <> ''
            AND NOT EXISTS (
                SELECT 1
                FROM user_contacts uc
                JOIN contacts c ON c.id = uc.contact_id
                WHERE 
                    uc.user_id = %s
                    AND c.phone = u.phone
            )
    """, (user_id, user_id))

    candidate_rows = cursor.fetchall()

    # Fetch current user's own vector
    cursor.execute("""
        SELECT vector_data
        FROM user_profile_embeddings
        WHERE user_id = %s
    """, (user_id,))

    current_user_row = cursor.fetchone()

    cursor.close()
    conn.close()

    if not candidate_rows or not current_user_row:
        return []

    # ---------------------------------------
    # 2) Prepare vectors
    # ---------------------------------------
    current_vector = np.array(json.loads(current_user_row["vector_data"])).reshape(1, -1)

    candidate_vectors = []
    candidate_ids = []
    user_meta = {}

    for row in candidate_rows:
        vector = json.loads(row["vector_data"])
        candidate_vectors.append(vector)
        candidate_ids.append(row["id"])

        user_meta[row["id"]] = {
            "fname": row["fname"],
            "lname": row["lname"],
            "phone": row["phone"]  # ✅ keep phone
        }

    candidate_vectors = np.array(candidate_vectors)

    # ---------------------------------------
    # 3) Compute similarity ONLY vs current user
    # ---------------------------------------
    similarities = cosine_similarity(current_vector, candidate_vectors)[0]

    MIN_SIMILARITY = 0.1
    recommendations = []

    for idx, sim_score in enumerate(similarities):
        if sim_score < MIN_SIMILARITY:
            continue

        other_user_id = candidate_ids[idx]

        recommendations.append({
            "user_id": other_user_id,
            "name": f"{user_meta[other_user_id]['fname']} {user_meta[other_user_id]['lname']}",
            "phone": user_meta[other_user_id]["phone"],  # ✅ return phone to frontend
            "similarity_score": float(sim_score)
        })

    recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
    return recommendations[:top_n]