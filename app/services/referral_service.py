from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_candidates
from app.services.llm_filter_service import llm_filter
from app.db import get_db


def referral_search(my_user_id: int, prompt: str):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    # 1️⃣ Get MY contacts who are also users AND allow referrals
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

    results = []

    # 2️⃣ Embed prompt ONCE
    query_embedding = get_embedding(prompt)

    # 3️⃣ For each referrer X → run SAME AI search logic as normal search
    for x in referrers:
        referrer_id = x["referrer_user_id"]

        # retrieve candidates FROM X's embeddings
        candidates = retrieve_candidates(
            user_id=referrer_id,
            query_embedding=query_embedding,
            top_k=20
        )

        if not candidates:
            continue

        # strict LLM filter (same as normal AI search)
        judged = llm_filter(prompt, candidates, top_n=1)

        # if X has at least ONE valid match → X can refer
        if judged:
            results.append({
                "name": x["display_name"],
                "phone": x["phone"]
            })

    return results
