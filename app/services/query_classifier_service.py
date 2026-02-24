import json
from app.core.config import client


def classify_query(prompt: str):

    system = """
You are a professional query analyzer.

Classify the query into ONE category:

1. broad_skill
   → General ability (e.g. "make applications", "programmer", "build websites")

2. specific_domain
   → Clear technical domain (e.g. Big Data, DevOps, Cybersecurity, Blockchain, AI)

3. personal_trait
   → Personality or soft skill (funny, trustworthy, friendly)

4. ambiguous
   → Unclear request

Return STRICT JSON:
{ "type": "<category>" }
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    data = json.loads(resp.choices[0].message.content)
    return data.get("type", "ambiguous")