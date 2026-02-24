from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_candidates
from app.services.llm_filter_service import llm_filter
from app.services.query_classifier_service import classify_query


def run_ai_search(user_id: int, prompt: str, top_k: int = 40, top_n: int = 5):

    # 1️⃣ Embed
    query_embedding = get_embedding(prompt)

    # 2️⃣ Retrieve
    candidates = retrieve_candidates(
        user_id=user_id,
        query_embedding=query_embedding,
        top_k=top_k
    )

    if not candidates:
        return []

    # 3️⃣ Classify
    query_type = classify_query(prompt)

    # 4️⃣ LLM filter
    judged = llm_filter(
        prompt=prompt,
        candidates=candidates,
        query_type=query_type,
        top_n=top_n
    )

    judged_map = {
        j["idx"]: j
        for j in judged
        if isinstance(j.get("idx"), int)
    }

    final = []

    # 5️⃣ STRICT MERGE (exact same as /search)
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

    # 6️⃣ Rank by confidence
    final.sort(key=lambda x: x["confidence"], reverse=True)

    return final[:top_n]