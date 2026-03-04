from fastapi import FastAPI
from app.routes.search import router as search_router
from app.routes.referral import router as referral_router
from app.routes import recommendation_routes
from app.routes.vector_routes import router as vector_router

app = FastAPI(title="AI Contact Search")

app.include_router(search_router)
app.include_router(referral_router)
app.include_router(recommendation_routes.router)
app.include_router(vector_router)