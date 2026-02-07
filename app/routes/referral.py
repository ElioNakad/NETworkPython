from fastapi import APIRouter
from app.models.schemas import ReferralSearchRequest
from app.services.referral_service import referral_search

router = APIRouter(prefix="/referral", tags=["Referral"])

@router.post("/search")
def search_referral(req: ReferralSearchRequest):
    return referral_search(req.user_id, req.prompt)
