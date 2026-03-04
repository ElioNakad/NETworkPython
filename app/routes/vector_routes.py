from fastapi import APIRouter
from app.services.vector_rebuild_service import rebuild_all_vectors

router = APIRouter()

@router.post("/rebuild-vectors")
def rebuild_vectors():
    return rebuild_all_vectors()