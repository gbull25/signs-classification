import base64
import logging
from datetime import datetime
from typing import Dict, List, Union

import redis
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .cropped_sign import CroppedSign

# from . import classify
from .model_loader import ModelLoader

MODELS = ModelLoader()
REDIS_CLIENT = redis.Redis(host='redis', port='6379', db=0)

app = FastAPI(
    # lifespan=lifespan,
    title="Predict the class of a sign",
    version="1.0",
    contact={
        "name": "Gleb Bulygin & Viktor Tikhomirov",
        "url": "https://github.com/gbull25/signs-classification"
    }
    )

templates = Jinja2Templates(directory="app/templates")


class ClassificationResult(BaseModel):
    sign_class: int | None = None
    sign_description: str | None = None
    message: str = "No prediction was made"


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
