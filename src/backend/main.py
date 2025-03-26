import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.backend.api.routes import router


app = FastAPI(title="Non-Vocal Sound Classifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("src.backend.main:app", host="0.0.0.0", port=8000, reload=True)
