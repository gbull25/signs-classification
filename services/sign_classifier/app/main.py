import logging
from typing import Dict, List, Union

import redis
import torch
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth.base_config import auth_backend, fastapi_users
from .auth.router import router as role_adding_router
from .auth.schemas import UserCreate, UserRead
from .cropped_sign import CroppedSign
from .model_loader import ModelLoader
from .pages.router import router as router_pages
from .rating.router import router as router_rating
from .utils import pool

# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.redis import RedisBackend
import io
import imageio
from imageio import v3 as iio
from fastapi import Response

# ---------------------------------------------------------------------------- #
#                       CLASSES, FUNCTIONS, GLOBAL VARS.                       #
# ---------------------------------------------------------------------------- #


class ClassificationResult(BaseModel):
    """Classification result validation model."""
    sign_class: int | None = None
    sign_description: str | None = None
    message: str = "No prediction was made"


class DetectionResult(BaseModel):
    """Classification result validation model."""
    signs_bboxes:  List | None = None
    #message: str = "No prediction was made"


def get_redis():
    """Get redis connection."""
    return redis.Redis(connection_pool=pool)


# load models as global var
MODELS = ModelLoader()


# ---------------------------------------------------------------------------- #
#                       DEFINE APP, MOUNT ROUTERS, ETC..                       #
# ---------------------------------------------------------------------------- #


app = FastAPI(
    # lifespaModelLoadern=lifespan,
    title="Predict the class of a sign",
    version="1.0",
    contact={
        "name": "Gleb Bulygin & Viktor Tikhomirov",
        "url": "https://github.com/gbull25/signs-classification"
    }
    )


# @app.on_event("startup")
# async def startup_event():
#     redis = aioredis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}", encoding="utf8", decode_responses=True)
#     FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")


templates = Jinja2Templates(directory="app/pages/templates")


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


# ---------------------------------------------------------------------------- #
#                                   ENDPOINTS                                  #
# ---------------------------------------------------------------------------- #


@app.get("/reload_models")
def reload_models() -> Dict[str, str]:
    global MODELS
    try:
        MODELS = ModelLoader()
        return {"message": "Success"}
    except Exception as e:
        return {"message": f"Failed with exception: {e}"}


@app.get(
        "/history",
        response_model=List[Dict[str, Union[str, int, None]]]
    )
def get_history(
    n_entries: int = 10,
    redis_conn=Depends(get_redis)
        ) -> List[Dict[str, Union[str, int, None]]] | Dict[str, str]:
    """
    Loads last {n_entries} processed images from redis db.

    Args:
        - n_entries: number of entries to return;
        - redis_conn: connection instance.

    Returns:
        - list of dicts with history entries;
        List[Dict[str, Union[str, int, None]]].
    """
    try:
        query = redis_conn.xrange("predictions:history", "-", "+", n_entries)
        logging.debug(f"Query received: {query}")
        logging.info(f"Successfully received history of {len(query)} items")
    except Exception as e:
        logging.info(f"An error occured during redis transaction: {e}")
        return {"message": f"Failed to load history: {e}"}

    # Make proper dict from query
    history_list = []
    for item in query:
        history_list.append(CroppedSign.from_redis(item[1]).to_html())

    return history_list


@app.get(
        "/history_pretty"
    )
def get_history_pretty(
    request: Request,
    n_entries: int = 10,
    redis_conn=Depends(get_redis)
        ):
    """
    Loads last {n_entries} processed images from redis db.

    Args:
        - n_entries: number of entries to return;
        - redis_conn: connection instance.

    Returns:
        - HTML-template to render for readability.
    """
    try:
        query = redis_conn.xrange("predictions:history", "-", "+", n_entries)
        logging.debug(f"Query received: {query}")
        logging.info(f"Successfully received history of {len(query)} items")
    except Exception as e:
        logging.info(f"An error occured during redis transaction: {e}")
        return {"message": f"Failed to load history: {e}"}

    # Make proper dict from query
    history_list = []
    for item in query:
        history_list.append(CroppedSign.from_redis(item[1]).to_html())

    return templates.TemplateResponse("history.html", {"request": request, "history_list": history_list})


@app.post(
    "/classify_sign",
    response_model=ClassificationResult
    )
def classify_sign(
    request: Request,
    file_img: UploadFile,
    model_name: str = 'cnn',
    redis_conn=Depends(get_redis)
        ) -> ClassificationResult:
    """Classify which sign is in the image.

    Args:
        - request: http request;
        - file_img: uploaded image of the sign;
        - model_name: name of the model to use for classification.

    Returns:
        - ClassificationResult: the result of the classification.
    """
    logging.info(f"Request received; host: {request.client.host}; "
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
    # TODO: fix this mess
    if model_name == 'sift':
        predicted_class = classify(MODELS.kmeans, MODELS.sift_model)
    else:
        predicted_class = classify(getattr(MODELS, f"{model_name}_model"))
    logging.info(f"Prediction results: {predicted_class}")

    # Write to redis history
    try:
        redis_conn.xadd(
            "predictions:history",
            sign.to_redis()
        )
        logging.info("Successfully added entry to the 'prediction:history'.")
        logging.info(f"Stream length: {redis_conn.xlen('predictions:history')}")
    except Exception as e:
        logging.error(f"An error occured during redis transaction: {e}")

    return ClassificationResult(
        message="Success",
        **predicted_class
    )


@app.post(
    "/detect_sign"
    )
def detect_sign(
    request: Request,
    file_img: UploadFile,
    model_name: str = 'yolo',
    redis_conn=Depends(get_redis)
        ):
    """Classify which sign is in the image.

    Args:
        - request: http request;
        - file_img: uploaded image of the sign;
        - model_name: name of the model to use for detection.

    Returns:
        - DetectionResult: the result of the detection.
    """
    logging.info(f"Request received; host: {request.client.host}; "
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

    detect = getattr(sign, f"detect_{model_name}")
    detections = detect(getattr(MODELS, f"{model_name}_model"))
    #logging.info(f"Prediction results: {sign_boxes}")
    #result = MODELS.yolo_model.predict(byte_img, conf=0.1)
    #logging.info(f"Classification results, classes: {result[0].boxes.cls}")
    #logging.info(f"Classification results, confidence: {result[0].boxes.conf}")
    #logging.info(f"Classification results, bboxes: {result[0].boxes.data}")

    # Return an image
    with io.BytesIO() as buf:
        iio.imwrite(buf, detections['crop_image'][0], plugin="pillow", format="JPEG")
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="test.jpeg"'}

    # Write to redis history
    try:
        redis_conn.xadd(
            "predictions:history",
            sign.to_redis()
        )
        logging.info("Successfully added entry to the 'prediction:history'.")
        logging.info(f"Stream length: {redis_conn.xlen('predictions:history')}")
    except Exception as e:
        logging.error(f"An error occured during redis transaction: {e}")

    #return FileResponse(detections['crop_image'])
    return Response(im_bytes, headers=headers, media_type='image/jpeg')


# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
