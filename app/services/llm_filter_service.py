import json
from app.core.config import client


def llm_filter(prompt: str, candidates: list, query_type: str, top_n: int = 5):

    packed = [
        {
            "idx": c["idx"],
            "name": c["name"],
            "profile_text": (c["profile_text"] or "")[:1500]
        }
        for c in candidates
    ]

    system = f"""
You are an intelligent contact matching AI.

The query type is: {query_type}

MATCHING RULES:

If query_type = broad_skill:
- Match general capability.
- General software engineers CAN match.
- Use overall technical richness.

If query_type = specific_domain:
- Require STRONG evidence that the contact works in that domain.
- Do NOT assume domain expertise from general software engineering.
- Domain must be clearly supported by labels, skills, or experience.

If query_type = personal_trait:
- Match personality descriptions or labels.

If query_type = ambiguous:
- Use best judgment.

Scoring:
0.9+  → Very strong match
0.75-0.9 → Strong match
0.6-0.75 → Possible match
Below 0.6 → EXCLUDE

Return STRICT JSON:
{{
  "results": [
    {{"idx": number, "confidence": number, "reason": string}}
  ]
}}
"""

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
        temperature=0
    )

    data = json.loads(resp.choices[0].message.content)
    results = data.get("results", [])

    # Hard exclude below 0.6
    return [r for r in results if r.get("confidence", 0) >= 0.6]