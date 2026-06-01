import json
from collections import deque

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.db import get_db


def get_default_labels(user_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT label
        FROM default_description
        WHERE users_id = %s
    """, (user_id,))

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return [row["label"] for row in rows]


def get_default_labels_with_cursor(cursor, user_id):
    cursor.execute("""
        SELECT label
        FROM default_description
        WHERE users_id = %s
    """, (user_id,))

    rows = cursor.fetchall()
    return [row["label"] for row in rows]


def build_path_node(row, display_name=None):
    full_name = f"{row['fname']} {row['lname']}".strip()

    return {
        "user_id": row["id"],
        "name": display_name if display_name else full_name,
        "full_name": full_name,
        "phone": row["phone"]
    }


def find_referral_path(cursor, start_user_id, target_user_id, max_depth=6):
    cursor.execute("""
        SELECT id, fname, lname, phone
        FROM users
        WHERE id = %s
    """, (start_user_id,))

    start_user = cursor.fetchone()

    if not start_user:
        return None

    start_node = build_path_node(start_user)
    queue = deque([(start_user_id, [start_node])])
    visited = {start_user_id}

    while queue:
        current_user_id, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        cursor.execute("""
            SELECT
                next_user.id,
                next_user.fname,
                next_user.lname,
                next_user.phone,
                uc.display_name
            FROM user_contacts uc
            JOIN contacts c
                ON c.id = uc.contact_id
            JOIN users next_user
                ON next_user.phone = c.phone
            JOIN users active_user
                ON active_user.id = %s
            LEFT JOIN contacts current_contact
                ON current_contact.phone = active_user.phone
            LEFT JOIN user_contacts blockcheck
                ON blockcheck.user_id = next_user.id
               AND blockcheck.contact_id = current_contact.id
               AND blockcheck.block = 'true'
            WHERE uc.user_id = %s
              AND next_user.refer = 'true'
              AND next_user.phone IS NOT NULL
              AND next_user.phone <> ''
              AND next_user.id != %s
              AND blockcheck.id IS NULL
        """, (current_user_id, current_user_id, current_user_id))

        neighbors = cursor.fetchall()

        for neighbor in neighbors:
            neighbor_id = neighbor["id"]

            if neighbor_id in visited:
                continue

            next_node = build_path_node(neighbor, neighbor.get("display_name"))
            next_path = path + [next_node]

            if neighbor_id == target_user_id:
                return next_path

            visited.add(neighbor_id)
            queue.append((neighbor_id, next_path))

    return None


def get_recommendations_for_user(user_id: int, top_n: int = 5):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # 1) Fetch embeddings excluding self and existing contacts.
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

    cursor.execute("""
        SELECT vector_data
        FROM user_profile_embeddings
        WHERE user_id = %s
    """, (user_id,))

    current_user_row = cursor.fetchone()

    if not candidate_rows or not current_user_row:
        cursor.close()
        conn.close()
        return []

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
            "phone": row["phone"]
        }

    candidate_vectors = np.array(candidate_vectors)
    similarities = cosine_similarity(current_vector, candidate_vectors)[0]

    min_similarity = 0.1
    recommendations = []

    for idx, sim_score in enumerate(similarities):
        if sim_score < min_similarity:
            continue

        other_user_id = candidate_ids[idx]

        recommendations.append({
            "user_id": other_user_id,
            "name": f"{user_meta[other_user_id]['fname']} {user_meta[other_user_id]['lname']}",
            "phone": user_meta[other_user_id]["phone"],
            "similarity_score": float(sim_score),
            "labels": get_default_labels_with_cursor(cursor, other_user_id)
        })

    recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
    recommendations = recommendations[:top_n]

    for recommendation in recommendations:
        path = find_referral_path(cursor, user_id, recommendation["user_id"])
        bridge = None

        if path and len(path) > 2:
            bridge = {
                "user_id": path[1]["user_id"],
                "name": path[1]["name"],
                "phone": path[1]["phone"]
            }

        recommendation["bridge_user"] = bridge
        recommendation["path"] = path

    cursor.close()
    conn.close()

    return recommendations
