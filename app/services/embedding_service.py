import numpy as np
from app.core.config import client

def get_embedding(text: str) -> np.ndarray:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding, dtype=np.float32)
