from fastapi import FastAPI
from app.routes.search import router as search_router

app = FastAPI(title="AI Contact Search")

app.include_router(search_router)
