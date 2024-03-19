import base64
import logging
from datetime import datetime
from typing import Dict, List, Union

import redis
from redis import asyncio as aioredis
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .cropped_sign import CroppedSign

# from . import classify
from .model_loader import ModelLoader

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

from .auth.base_config import auth_backend, fastapi_users
from .auth.schemas import UserCreate, UserRead
from .auth.config import REDIS_HOST, REDIS_PORT
from .auth.router import router as role_adding_router
from .pages.router import router as router_pages
from .rating.router import router as router_rating

MODELS = ModelLoader()
REDIS_CLIENT = redis.Redis(host='redis', port='5370', db=0)

app = FastAPI(
    # lifespan=lifespan,
    title="Predict the class of a sign",
    version="1.0",
    contact={
        "name": "Gleb Bulygin & Viktor Tikhomirov",
        "url": "https://github.com/gbull25/signs-classification"
    }
    )


templates = Jinja2Templates(directory="app/pages/templates")


class ClassificationResult(BaseModel):
    sign_class: int | None = None
    sign_description: str | None = None
    message: str = "No prediction was made"


app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["Auth"],
)

app.include_router(router_rating)
app.include_router(router_pages)
app.include_router(role_adding_router)

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["Content-Type", "Set-Cookie", "Access-Control-Allow-Headers", "Access-Control-Allow-Origin",
                   "Authorization"],
)


@app.on_event("startup")
async def startup_event():
    redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


@app.get("/reload_models")
def reload_models():
    global MODELS
    MODELS = ModelLoader()


@app.get("/history", response_model=List[Dict[str, Union[str, int, None]]])
def get_history(n_entries: int = 10) -> List[Dict[str, Union[str, int, None]]]:
    try:
        query = REDIS_CLIENT.xrange("predictions:history", "-", "+", n_entries)
        logging.debug(f"Query received: {query}")
        logging.info(f"Successfully received history of {len(query)} items")
    except Exception as e:
        logging.info(f"An error occured during redis transaction: {e}")

    # Make proper dict from query
    history_list = []
    for item in query:       
        history_list.append(CroppedSign.from_redis(item[1]).to_html())

    return history_list

@app.get("/history_pretty")
def get_history_pretty(request: Request, n_entries: int = 10):
    try:
        query = REDIS_CLIENT.xrange("predictions:history", "-", "+", n_entries)
        logging.debug(f"Query received: {query}")
        logging.info(f"Successfully received history of {len(query)} items")
    except Exception as e:
        logging.info(f"An error occured during redis transaction: {e}")

    # Make proper dict from query
    history_list = []
    for item in query:
        history_list.append(CroppedSign.from_redis(item[1]).to_html())

    return templates.TemplateResponse("history.html", {"request": request, "history_list": history_list})


@app.post("/classify_sign", response_model=ClassificationResult)
def classify_sign(request: Request, file_img: UploadFile, model_name: str = 'cnn') -> ClassificationResult:
    """Classify which sign is in the image.
    
    Args:
        - request: http request;
        - file_img: uploaded image of the sign;
        - model_name: name of the model to use for classification.
    
    Returns:
        - ClassificationResult: the result of the classification.
    """
    logging.info(f"Request received; host: {request.client.host}; " \
                 f"filename: {file_img.filename}, content_type: {file_img.content_type}")
    
    # Read file
    try:
        byte_img = file_img.file.read()
    except Exception as e:
        logging.error(f"Unexpected error while reading uploaded file:\n\n {e}")
        return ClassificationResult(message="Unexpected error while reading uploaded file")

    # Define CroppedSign instance
    sign = CroppedSign(
        img=byte_img,
        filename=file_img.filename
    )

    # Access attribute depending on passed model name
    classify = getattr(sign, f"classify_{model_name}")
    predicted_class = classify(getattr(MODELS, f"{model_name}_model"))
    logging.info(f"Prediction results: {predicted_class}")

    # Write to redis history
    try:
        REDIS_CLIENT.xadd(
            "predictions:history",
            sign.to_redis()
        )
        logging.info(f"Successfully added entry to the 'prediction:history'.")
        logging.info(f"Stream length: {REDIS_CLIENT.xlen('predictions:history')}")
    except Exception as e:
        logging.error(f"An error occured during redis transaction: {e}")

    return ClassificationResult(
        message="Success",
        **predicted_class
    )


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
