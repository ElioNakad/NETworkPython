from fastapi import FastAPI
from app.routes.search import router as search_router
from app.routes.referral import router as referral_router

app = FastAPI(title="AI Contact Search")

app.include_router(search_router)
app.include_router(referral_router)
