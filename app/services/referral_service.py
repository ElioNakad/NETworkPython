from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_candidates
from app.services.llm_filter_service import llm_filter
from app.services.query_classifier_service import classify_query
import mysql.connector
import os


def referral_search(my_user_id: int, prompt: str):

    # 🔥 Open fresh connection every time
    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True   # VERY IMPORTANT
    )

    cursor = db.cursor(dictionary=True)

    # 1️⃣ Get my contacts who are users AND allow referral
    cursor.execute("""
        SELECT
            uc.display_name,
            u.id   AS referrer_user_id,
            u.phone,
            u.refer
        FROM user_contacts uc
        JOIN contacts c ON c.id = uc.contact_id
        JOIN users u ON u.phone = c.phone
        WHERE uc.user_id = %s
          AND u.refer = 'true'
    """, (my_user_id,))

    referrers = cursor.fetchall()

    if not referrers:
        cursor.close()
        db.close()
        return []

    # 2️⃣ Embed & classify once
    query_embedding = get_embedding(prompt)
    query_type = classify_query(prompt)

    results = []

    # 3️⃣ For each referrer → run SAME search pipeline
    for x in referrers:

        referrer_id = x["referrer_user_id"]

        candidates = retrieve_candidates(
            user_id=referrer_id,
            query_embedding=query_embedding,
            top_k=40
        )

        if not candidates:
            continue

        judged = llm_filter(
            prompt=prompt,
            candidates=candidates,
            query_type=query_type
        )

        if not judged:
            continue

        judged_map = {
            j["idx"]: j
            for j in judged
            if isinstance(j.get("idx"), int)
        }

        final = []

        for c in candidates:
            j = judged_map.get(c["idx"])
            if not j:
                continue

            final.append({
                "confidence": j.get("confidence", 0.0)
            })

        if not final:
            continue

        best_confidence = max(f["confidence"] for f in final)

        results.append({
            "name": x["display_name"],
            "phone": x["phone"],
            "confidence": best_confidence
        })

    # 4️⃣ Rank by strength
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # 🔥 CLOSE CONNECTION
    cursor.close()
    db.close()

    return results[:5]