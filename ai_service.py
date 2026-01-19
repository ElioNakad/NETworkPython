from fastapi import FastAPI
from pydantic import BaseModel
import os, json, numpy as np, mysql.connector
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# SETUP
# =========================
load_dotenv()
app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

db = mysql.connector.connect(
    host=os.getenv("DB_HOST"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    database=os.getenv("DB_NAME"),
)
cur = db.cursor(dictionary=True)

# =========================
# MODELS
# =========================
class SearchRequest(BaseModel):
    user_id: int
    prompt: str

# =========================
# HELPERS
# =========================
def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def get_embedding(text: str):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32)

def llm_filter(prompt: str, candidates: list, top_n: int = 5):
    """
    candidates: list of dicts:
      { "idx": int, "name": str, "phone": str, "profile_text": str, "score": float }
    returns: list of dicts:
      { "idx": int, "match": bool, "reason": str, "confidence": float }
    """

    # Keep the context small & clean
    packed = []
    for c in candidates:
        text = (c.get("profile_text") or "").strip()
        if not text:
            text = "(no profile text)"
        packed.append({
            "idx": c["idx"],
            "name": c["name"],
            "phone": c["phone"],
            "profile_text": text[:800]  # safety trim
        })

    system = (
        "You are a strict filter. The user wrote a query about people in their contacts.\n"
        "Decide which candidates truly match the intent of the user query.\n"
        "IMPORTANT:\n"
        "- If the query implies negative sentiment (hate, dislike, avoid), do NOT select people described positively.\n"
        "- If there is not enough evidence, mark match=false.\n"
        "- Return JSON only."
    )

    user = {
        "query": prompt,
        "candidates": packed,
        "instructions": {
            "return_format": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "idx": {"type": "integer"},
                                "match": {"type": "boolean"},
                                "confidence": {"type": "number"},
                                "reason": {"type": "string"}
                            },
                            "required": ["idx", "match", "confidence", "reason"]
                        }
                    }
                },
                "required": ["results"]
            },
            "max_selected": top_n
        }
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    data = json.loads(resp.choices[0].message.content)
    return data.get("results", [])

# =========================
# ROUTES
# =========================
@app.post("/search")
def search(req: SearchRequest):
    q = get_embedding(req.prompt)

    cur.execute("""
        SELECT uce.contact_id, uce.embedding, uce.profile_text,
               uc.display_name, c.phone
        FROM user_contact_embeddings uce
        JOIN contacts c ON c.id = uce.contact_id
        LEFT JOIN user_contacts uc
          ON uc.user_id = uce.user_id
         AND uc.contact_id = uce.contact_id
        WHERE uce.user_id = %s
    """, (req.user_id,))

    raw = []
    for r in cur.fetchall():
        emb = np.array(json.loads(r["embedding"]), dtype=np.float32)
        score = cosine(q, emb)

        raw.append({
            "contact_id": r["contact_id"],
            "name": r["display_name"] or "(no name)",
            "phone": r["phone"],
            "profile_text": r["profile_text"],
            "score": score
        })

    # 1) retrieval: grab more candidates
    raw.sort(key=lambda x: x["score"], reverse=True)
    candidates = raw[:20]  # TOP_K_RETRIEVE

    # add idx for LLM reference
    for i, c in enumerate(candidates):
        c["idx"] = i

    # 2) generation/reasoning: filter with LLM
    judged = llm_filter(req.prompt, candidates, top_n=5)

    # build final list in ranked order by confidence then embedding score
    judged_map = {j["idx"]: j for j in judged if isinstance(j.get("idx"), int)}

    final = []
    for c in candidates:
        j = judged_map.get(c["idx"])
        if not j:
            continue
        if j.get("match") is True:
            final.append({
                "name": c["name"],
                "phone": c["phone"],
                "score": c["score"],
                "confidence": float(j.get("confidence", 0)),
                "reason": j.get("reason", ""),
                "profile_text": c["profile_text"]
            })

    final.sort(key=lambda x: (x["confidence"], x["score"]), reverse=True)
    return final[:5]
