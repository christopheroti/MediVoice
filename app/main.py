from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TriageRequest, TriageResponse
from app import triage_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    triage_service.load_model()  # load model once at startup
    yield


app = FastAPI(title="MediVoice AI API", version="1.0", lifespan=lifespan)

# Only these websites may call the API from a browser.
ALLOWED_ORIGINS = [
    "https://medical-ai-gamma.vercel.app",  # production frontend (Vercel)
    "http://localhost:3000",                # local dev (React/Next.js)
    "http://localhost:5173",                # local dev (Vite)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "MediVoice AI API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
def triage(request: TriageRequest):
    try:
        return triage_service.predict(request.text, request.chief_complaint or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
