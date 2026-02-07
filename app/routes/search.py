from fastapi import APIRouter
from app.models.schemas import SearchRequest
from app.services.embedding_service import get_embedding
from app.services.retrieval_service import retrieve_candidates
from app.services.llm_filter_service import llm_filter

router = APIRouter()

@router.post("/search")
def search(req: SearchRequest):
    #tranforming the prompt to an embedding 
    q = get_embedding(req.prompt)
    #fetche all the candidates for this prompt
    candidates = retrieve_candidates(req.user_id, q)
    #send the query+profile_text to the llm so it filters and rank them
    judged = llm_filter(req.prompt, candidates)
    judged_map = {j["idx"]: j for j in judged if isinstance(j.get("idx"), int)}

    final = []
    for c in candidates:
        j = judged_map.get(c["idx"])
        if not j:
            continue
        final.append({
            "name": c["name"],
            "phone": c["phone"],
            "score": c["score"],
            "confidence": j.get("confidence", 0),
            "reason": j.get("reason", ""),
            "profile_text": c["profile_text"]
        })

    final.sort(key=lambda x: (x["confidence"], x["score"]), reverse=True)
    return final[:5]
