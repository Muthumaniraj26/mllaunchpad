# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from huggingface_hub import list_models, list_datasets
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 1. Initialize the FastAPI App ---
app = FastAPI()

# --- 2. Configure CORS ---
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Load AI Models ---
print("Loading task identification model...")
try:
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    print("✅ FLAN-T5 model loaded successfully!")
except Exception as e:
    print(f"🔥 Error loading FLAN-T5 model: {e}")
    generator = None

# Optional fallback: Zero-shot classifier if FLAN-T5 fails
try:
    print("Loading fallback zero-shot model...")
    fallback_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("✅ Fallback model loaded successfully!")
except Exception as e:
    print(f"🔥 Error loading fallback model: {e}")
    fallback_classifier = None

# --- 4. Define Data Models ---
class AnalysisRequest(BaseModel):
    problem_statement: str

class AnalysisResponse(BaseModel):
    identified_task: str
    confidence_score: float
    suggested_models: list
    suggested_datasets: list

# --- 5. Root Endpoint ---
@app.get("/")
def read_root():
    return {"message": "✅ ML Launchpad API is running with FLAN-T5!"}

# --- 6. Analyze Endpoint ---
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_problem(request: AnalysisRequest):
    problem_text = request.problem_statement.strip()

    if generator:
        # Use FLAN-T5 for task detection
        prompt = f"Identify the most suitable machine learning task for the following problem: {problem_text}. Return only the task name."
        try:
            response = generator(prompt, max_new_tokens=30)[0]['generated_text'].strip()
            task = response.lower().replace(".", "")
            confidence = 0.9  # heuristic for FLAN-T5
        except Exception as e:
            print(f"⚠️ FLAN-T5 failed: {e}")
            task, confidence = None, 0.0
    else:
        task, confidence = None, 0.0

    # Fallback to zero-shot if FLAN-T5 fails
    if not task and fallback_classifier:
        candidate_labels = [
            "text classification", "text generation", "summarization", "translation",
            "image classification", "object detection", "speech recognition",
            "question answering", "named entity recognition", "sentiment analysis",
            "tabular regression", "reinforcement learning", "time series forecasting"
        ]
        result = fallback_classifier(problem_text, candidate_labels)
        task = result['labels'][0]
        confidence = result['scores'][0]

    # If still no task, raise error
    if not task:
        raise HTTPException(status_code=500, detail="Could not identify task.")

    # Fetch top models & datasets for the identified task
    try:
        model_ids = [model.modelId for model in list_models(filter=task, sort="downloads", direction=-1, limit=5)]
    except:
        model_ids = []

    try:
        dataset_ids = [d.id for d in list_datasets(search=task, sort="downloads", direction=-1, limit=5)]
    except:
        dataset_ids = []

    return AnalysisResponse(
        identified_task=task,
        confidence_score=confidence,
        suggested_models=model_ids,
        suggested_datasets=dataset_ids
    )

# --- 7. Run the Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
