from fastapi import APIRouter, Query
from app.services.contact_vector_rebuild_service import (
    rebuild_contact_embedding,
    rebuild_dirty_contact_embeddings,
)
from app.services.vector_rebuild_service import rebuild_all_vectors

router = APIRouter()

@router.post("/rebuild-vectors")
def rebuild_vectors():
    return rebuild_all_vectors()

@router.post("/rebuild-contact-vector/{embedding_id}")
def rebuild_contact_vector(embedding_id: int):
    return rebuild_contact_embedding(embedding_id)

@router.post("/rebuild-dirty-contact-vectors")
def rebuild_dirty_contact_vectors(
    limit: int = Query(100, ge=1, le=500),
    embedding_ids: list[int] | None = Query(None)
):
    return rebuild_dirty_contact_embeddings(limit=limit, embedding_ids=embedding_ids)
