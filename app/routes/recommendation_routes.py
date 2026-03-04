from fastapi import APIRouter
from app.services.recommendation_service import get_recommendations_for_user

router = APIRouter()


@router.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 5):
    return get_recommendations_for_user(user_id, top_n)