import logging
import pathlib
import shutil
import uuid
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union

import redis
from fastapi import Depends, FastAPI, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from imageio import v3 as iio
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert

from .auth.base_config import auth_backend, fastapi_users
from .auth.database import get_async_session
from .auth.models import results
from .auth.router import router as role_adding_router
from .auth.schemas import UserCreate, UserRead
from .cropped_sign import CroppedSign
from .model_loader import ModelLoader
from .pages.router import router as router_pages
from .rating.router import router as router_rating
from .sign_detection import SignDetection
from .utils import pool

# from fastapi_cache import FastAPICache
# from fastapi_cache.backends.redis import RedisBackend


# ---------------------------------------------------------------------------- #
#                       CLASSES, FUNCTIONS, GLOBAL VARS.                       #
# ---------------------------------------------------------------------------- #


class ClassificationResult(BaseModel):
    """Classification result validation model."""
    user_id: str = "0"
    result_filepath: str | None = None
    detection_id: int | None = None
    detection_conf: float = 0.0
    sign_class: int | None = None
    sign_description: str | None = None
    bbox: str | None = None
    frame_number: int = 1
    detection_speed: float = 0.0
    model_used: str = "cnn"


class DetectionResult(BaseModel):
    """Classification result validation model."""
    signs_bboxes:  List | None = None
    #message: str = "No prediction was made"


def get_redis():
    """Get redis connection."""
    return redis.Redis(connection_pool=pool)


def make_user_id():
    """Generate unique user id"""
    return str(uuid.uuid4())


async def write_results(classification_result: ClassificationResult, session):
    insert_stmt = insert(results).values(**classification_result.dict())
    do_update_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

    await session.execute(do_update_stmt)
    await session.commit()
    return {"status": "success"}


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
        img= byte_img,
        id = 0,
        filename=file_img.filename
    )

    # Access attribute depending on passed model name
    model_name = "cnn"
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
    "/detect_and_classify_signs"
    )
async def detect_and_classify_signs(
    request: Request,
    file_data: UploadFile,
    suffix: str = "_",
    user_id="0",
    redis_conn=Depends(get_redis),
    postgres_session=Depends(get_async_session)
        ) -> List[ClassificationResult]:
    """Classify which sign is in the image.

    Args:
        - request: http request;
        - file_img: uploaded image of the sign;
        - model_name: name of the model to use for detection.

    Returns:
        - DetectionResult: the result of the detection.
    """
    if user_id == "0":
        user_id = make_user_id()
    logging.info(f"Request received; host: {request.client.host}; "
                 f"filename: {file_data.filename}, content_type: {file_data.content_type}")

    # Copy file to named tmp file
    # https://stackoverflow.com/a/63581187
    if suffix == "_":
       suffix = pathlib.Path(file_data.filename).suffix

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file_data.file, tmp)
            tmp_path = pathlib.Path(tmp.name)
    except Exception as e:
        logging.error(f"Unexpected error while managing uploaded file:\n\n {e}")
        return ClassificationResult(message="Unexpected error while reading uploaded file")
    finally:
        file_data.file.close()
    
    data = SignDetection(tmp_path, user_id, MODELS.yolo_model)
    data.detect()
    classification_results = []

    # # Generate cropped_signs from detected objects
    
    for id, obj in data.stream_objects():
        sign = CroppedSign(
            user_id=user_id,
            source_filepath=obj["file_path"],
            img=obj["cropped_img"],
            bbox=obj["bbox"],
            id=id,
            frame_number=obj["frame_number"],
            detection_speed=obj["detection_speed"]
            # filename=obj["filename"]
        )
        # Classify sign using cnn
        predicted_class = sign.classify_cnn(MODELS.cnn_model)
        logging.info(f"Prediction results: {predicted_class}")

        res = ClassificationResult(**predicted_class)
        await write_results(res, postgres_session)
        classification_results.append(res)

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

    return classification_results


# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
