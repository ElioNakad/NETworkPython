from fastapi import APIRouter
from app.models.schemas import SearchRequest
from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_candidates
from app.services.llm_filter_service import llm_filter
from app.services.query_classifier_service import classify_query

import mysql.connector
import os

router = APIRouter()


@router.post("/search")
def search(req: SearchRequest):

    # 🔥 Dedicated DB connection (like referral_search)
    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        autocommit=True
    )

    cursor = db.cursor(dictionary=True)

    # 1️⃣ Embed query
    query_embedding = get_embedding(req.prompt)

    # 2️⃣ Retrieve candidates (semantic stage)
    candidates = retrieve_candidates(
        req.user_id,
        query_embedding,
        top_k=40
    )

    if not candidates:
        cursor.close()
        db.close()
        return []

    # 3️⃣ Get my phone
    cursor.execute(
        "SELECT phone FROM users WHERE id = %s",
        (req.user_id,)
    )

    my_phone = cursor.fetchone()["phone"]

    # 4️⃣ Find users who BLOCKED me
    cursor.execute("""
        SELECT u.phone
        FROM users u
        JOIN contacts c
            ON c.phone = %s
        JOIN user_contacts uc
            ON uc.user_id = u.id
           AND uc.contact_id = c.id
           AND uc.block = 'true'
    """, (my_phone,))

    blocked_phones = {r["phone"] for r in cursor.fetchall()}

    # 5️⃣ Remove blocked candidates
    candidates = [
        c for c in candidates
        if c["phone"] not in blocked_phones
    ]

    if not candidates:
        cursor.close()
        db.close()
        return []

    # 6️⃣ Classify query
    query_type = classify_query(req.prompt)

    # 7️⃣ LLM filtering
    judged = llm_filter(
        prompt=req.prompt,
        candidates=candidates,
        query_type=query_type
    )

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
            "name": c["name"],
            "phone": c["phone"],
            "confidence": j.get("confidence", 0.0),
            "reason": j.get("reason", ""),
            "profile_text": c["profile_text"]
        })

    # 8️⃣ Rank by LLM confidence
    final.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    cursor.close()
    db.close()

    return final[:5]