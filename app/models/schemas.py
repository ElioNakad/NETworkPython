from pydantic import BaseModel

class SearchRequest(BaseModel):
    user_id: int
    prompt: str

class ReferralSearchRequest(BaseModel):
    user_id: int
    prompt: str