import json
from app.core.config import client

def llm_filter(prompt: str, candidates: list, top_n: int = 5):
    packed = [
        {
            "idx": c["idx"],
            "name": c["name"],
            "phone": c["phone"],
            "profile_text": (c["profile_text"] or "")[:800]
        }
        for c in candidates
    ]

    system = (
        "You are a strict filter for contact search.\n"
        "Return JSON only."
    )

    user = {
        "query": prompt,
        "candidates": packed,
        "max_selected": top_n
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    return json.loads(resp.choices[0].message.content).get("results", [])
