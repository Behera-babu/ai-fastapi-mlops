import os
import time
import uuid
import logging
import asyncio
from typing import Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Security, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ML imports
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
API_KEY = os.getenv("API_KEY", "your-secret-api-key-change-in-production")

# Initialize FastAPI
app = FastAPI(
    title="AI FastAPI MLOps",
    description="Production-ready AI service with SOTA models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - FIXED SECURITY
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency', ['method', 'endpoint'])
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time', ['model_name'])
ACTIVE_REQUESTS = Gauge('api_active_requests', 'Number of active requests')
MODEL_LOADED = Gauge('model_loaded', 'Model load status', ['model_name'])

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints"""
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )

# Global model variables
vision_model = None
vision_processor = None
nlp_classifier = None

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class or result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model: str = Field(..., description="Model used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    top_predictions: Optional[list] = Field(None, description="Top predictions with scores")

class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text")
    task: str = Field(default="sentiment", description="Task type: sentiment, classification")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('Text cannot be empty')
        return v.strip()

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    models_loaded: dict

# Model loading
async def load_vision_model():
    """Load Vision Transformer model"""
    global vision_model, vision_processor
    try:
        logger.info("Loading Vision Transformer model...")
        vision_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        vision_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        vision_model.eval()
        MODEL_LOADED.labels(model_name='vit-base-patch16-224').set(1)
        logger.info("✅ Vision model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load vision model: {str(e)}")
        MODEL_LOADED.labels(model_name='vit-base-patch16-224').set(0)

async def load_nlp_model():
    """Load NLP sentiment model"""
    global nlp_classifier
    try:
        logger.info("Loading NLP sentiment model...")
        nlp_classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        MODEL_LOADED.labels(model_name='distilbert-sentiment').set(1)
        logger.info("✅ NLP model loaded!")
    except Exception as e:
        logger.error(f"❌ Failed to load NLP model: {str(e)}")
        MODEL_LOADED.labels(model_name='distilbert-sentiment').set(0)

@app.on_event("startup")
async def startup_event():
    """Application startup - load models"""
    logger.info("🚀 Starting AI FastAPI MLOps service...")
    try:
        await asyncio.gather(load_vision_model(), load_nlp_model())
        logger.info("✅ All models loaded! Service ready!")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("👋 Shutting down...")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/', status='200').inc()
    return {
        "message": "AI FastAPI MLOps Service",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return HealthResponse(
        status="healthy" if all([vision_model, nlp_classifier]) else "degraded",
        timestamp=time.time(),
        version="2.0.0",
        models_loaded={"vision": vision_model is not None, "nlp": nlp_classifier is not None}
    )

@app.post("/predict/vision", response_model=PredictionResponse)
@limiter.limit("10/minute")
async def predict_vision(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Vision model inference with REAL ViT model"""
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    ACTIVE_REQUESTS.inc()
    
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        image = Image.open(BytesIO(contents)).convert('RGB')
        
        if not vision_model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        inputs = vision_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = vision_model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_idx = outputs.logits.argmax(-1).item()
        confidence = probabilities[0][predicted_idx].item()
        predicted_label = vision_model.config.id2label[predicted_idx]
        
        inference_time = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/vision', status='200').inc()
        
        return PredictionResponse(
            prediction=predicted_label,
            confidence=round(confidence, 4),
            model="vit-base-patch16-224",
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )
    finally:
        ACTIVE_REQUESTS.dec()

@app.post("/predict/nlp", response_model=PredictionResponse)
@limiter.limit("20/minute")
async def predict_nlp(
    nlp_request: NLPRequest,
    api_key: str = Depends(verify_api_key)
):
    """NLP inference with REAL DistilBERT model"""
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    ACTIVE_REQUESTS.inc()
    
    try:
        if not nlp_classifier:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        result = nlp_classifier(nlp_request.text)[0]
        inference_time = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/nlp', status='200').inc()
        
        return PredictionResponse(
            prediction=result['label'],
            confidence=round(result['score'], 4),
            model="distilbert-sentiment",
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )
    finally:
        ACTIVE_REQUESTS.dec()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
import uuid
import logging
import os
from io import BytesIO
from PIL import Image
from typing import Optional
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency', ['endpoint'])
ERROR_COUNT = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])

# Get environment variables
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

app = FastAPI(
    title="AI FastAPI MLOps",
    description="Production-ready AI service with real ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - FIXED SECURITY
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Fixed: specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Fixed: specific methods only
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# Global model variables
vision_model = None
vision_processor = None
nlp_pipeline = None

# Pydantic models
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class or result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model: str = Field(..., description="Model used for prediction")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")

class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text")
    task: str = Field(default="sentiment", description="Task type: sentiment, classification")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError('Text cannot be empty')
        return v.strip()

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    models_loaded: bool
    environment: str

# API Key authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

# Startup time
startup_time = time.time()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint - API information"""
    REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    return {
        "message": "AI FastAPI MLOps Service",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "environment": ENVIRONMENT
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes"""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="2.0.0",
        models_loaded=vision_model is not None and nlp_pipeline is not None,
        environment=ENVIRONMENT
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict/vision", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_vision(
    file: UploadFile = File(..., description="Image file for classification")
):
    """
    Vision model inference endpoint with REAL ViT model.
    Requires API key authentication.
    """
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    try:
        REQUEST_COUNT.labels(endpoint="/predict/vision", method="POST").inc()
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            ERROR_COUNT.labels(endpoint="/predict/vision", error_type="invalid_content_type").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            ERROR_COUNT.labels(endpoint="/predict/vision", error_type="empty_file").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            ERROR_COUNT.labels(endpoint="/predict/vision", error_type="file_too_large").inc()
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 10MB limit"
            )
        
        # Validate and process image
        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
            logger.info(f"Processing image: {file.filename}, size: {len(contents)} bytes, format: {image.format}")
        except Exception as e:
            ERROR_COUNT.labels(endpoint="/predict/vision", error_type="invalid_image").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Real model inference
        if vision_model is None or vision_processor is None:
            ERROR_COUNT.labels(endpoint="/predict/vision", error_type="model_not_loaded").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vision model not loaded. Service is initializing."
            )
        
        # Process image with ViT
        inputs = vision_processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = vision_model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        # Get class name
        predicted_label = vision_model.config.id2label[predicted_class_idx]
        
        inference_time = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict/vision").observe(inference_time / 1000)
        
        response = PredictionResponse(
            prediction=predicted_label,
            confidence=round(confidence, 4),
            model="google/vit-base-patch16-224",
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )
        
        logger.info(f"Vision prediction completed: {request_id}, prediction: {predicted_label}, confidence: {confidence:.4f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict/vision", error_type="internal_error").inc()
        logger.error(f"Error in vision prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during prediction: {str(e)}"
        )

@app.post("/predict/nlp", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict_nlp(request: NLPRequest):
    """
    NLP model inference endpoint with REAL sentiment analysis model.
    Requires API key authentication.
    """
    start_time = time.time()
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    try:
        REQUEST_COUNT.labels(endpoint="/predict/nlp", method="POST").inc()
        logger.info(f"Processing NLP request: task={request.task}, text_length={len(request.text)}")
        
        if nlp_pipeline is None:
            ERROR_COUNT.labels(endpoint="/predict/nlp", error_type="model_not_loaded").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NLP model not loaded. Service is initializing."
            )
        
        # Real model inference
        result = nlp_pipeline(request.text)[0]
        
        inference_time = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict/nlp").observe(inference_time / 1000)
        
        response = PredictionResponse(
            prediction=result['label'],
            confidence=round(result['score'], 4),
            model="distilbert-base-uncased-finetuned-sst-2-english",
            inference_time_ms=round(inference_time, 2),
            request_id=request_id
        )
        
        logger.info(f"NLP prediction completed: {request_id}, prediction: {result['label']}, confidence: {result['score']:.4f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict/nlp", error_type="internal_error").inc()
        logger.error(f"Error in NLP prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during prediction: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Application startup tasks - Load ML models"""
    global vision_model, vision_processor, nlp_pipeline
    
    logger.info("Starting AI FastAPI MLOps service...")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info("Loading ML models...")
    
    try:
        # Load Vision Transformer model
        logger.info("Loading ViT model for image classification...")
        vision_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        vision_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        vision_model.eval()  # Set to evaluation mode
        logger.info("✅ ViT model loaded successfully!")
        
        # Load NLP sentiment analysis model
        logger.info("Loading sentiment analysis model...")
        nlp_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logger.info("✅ NLP model loaded successfully!")
        
        logger.info("🚀 All models loaded! Service ready!")
        logger.info(f"⚡ Startup time: {time.time() - startup_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        logger.error("Service will start but predictions will fail until models are loaded.")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down AI FastAPI MLOps service...")
    logger.info("Cleanup complete!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
