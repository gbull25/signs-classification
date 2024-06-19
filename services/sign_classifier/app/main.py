import base64
import logging
import pathlib
import shutil
import uuid
from tempfile import NamedTemporaryFile
from typing import Dict, List, Union

import folium
import pandas as pd
import redis
from fastapi import Depends, FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

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
    classification_speed: float = 0.0
    model_used: str = "cnn"


class DetectionResult(BaseModel):
    """Detection result validation model."""
    signs_bboxes:  List | None = None


def get_redis():
    """Get redis connection."""
    return redis.Redis(connection_pool=pool)


def make_user_id():
    """Generate unique user id"""
    return str(uuid.uuid4())


async def write_results(
        classification_result: List[ClassificationResult],
        session: AsyncSession,
        len_threshold: int = 1000
        ) -> Dict[str, Union[bool, str]]:
    """Write classification results to PostgreSQL database.

    Args:
        classification_result (List[ClassificationResult]): list with classification results.
        session (AsyncSession): database connection session.
        len_threshold (int, optional): Threshold after which batch writing should be carried out. Defaults to 1000.

    Returns:
        Dict[str, Union[bool, str]]: result of the operation whether if was successful or not.
    """
    classification_result = [entry.dict() for entry in classification_result]
    try:
        if len(classification_result) > len_threshold:
            num_batches = len(classification_result) / len_threshold
            start = 0
            stop = len_threshold
            for _ in range(int(num_batches)):
                batch = classification_result[start:stop]
                start = stop
                stop += len_threshold

                insert_stmt = insert(results).values(batch)
                do_update_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

                await session.execute(do_update_stmt)
                await session.commit()

            if num_batches % 1 != 0:
                last_batch = classification_result[start:]
                insert_stmt = insert(results).values(last_batch)
                do_update_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

                await session.execute(do_update_stmt)
                await session.commit()
            return {"success": True, "message": "ok"}
        else:
            insert_stmt = insert(results).values(classification_result)
            do_update_stmt = insert_stmt.on_conflict_do_nothing(index_elements=["id"])

            await session.execute(do_update_stmt)
            await session.commit()
            return {"success": True, "message": "ok"}

    except Exception as e:
        return {"success": False, "message": f"Failed with exception: {e}"}


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
def reload_models() -> Dict[str, Union[bool, str]]:
    """Reload models to RAM.

    Returns:
        Dict[str, Union[bool, str]]: result of the operation whether if was successful or not.
    """
    global MODELS
    try:
        MODELS = ModelLoader()
        return {"success": True, "message": "ok"}
    except Exception as e:
        return {"success": f"Failed with exception: {e}"}


@app.get(
        "/history",
        response_model=List[Dict[str, Union[str, int, None]]]
    )
def get_history(
    n_entries: int = 10,
    redis_conn: redis.Redis = Depends(get_redis)
        ) -> List[Dict[str, Union[str, int, None]]] | Dict[str, str]:
    """Loads last {n_entries} processed images from redis db.

    Args:
        n_entries (int, optional): number of entries to load. Defaults to 10.
        redis_conn (redis.Redis, optional): redis connection. Defaults to Depends(get_redis).

    Returns:
        List[Dict[str, Union[str, int, None]]] | Dict[str, str]: list of dicts with history entries
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


@app.get("/history_pretty")
def get_history_pretty(
    request: Request,
    n_entries: int = 10,
    redis_conn: redis.Redis = Depends(get_redis)
        ) -> Jinja2Templates.TemplateResponse:
    """Loads and renders to html last {n_entries} processed images from redis db.

    Args:
        request (Request): request instance.
        n_entries (int, optional): number of entries to load. Defaults to 10.
        redis_conn (redis.Redis, optional): redis connection. Defaults to Depends(get_redis).

    Returns:
        Jinja2Templates.TemplateResponse: html render.
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
    logging.error(history_list)

    return templates.TemplateResponse("history.html", {"request": request, "history_list": history_list})


@app.post(
    "/classify_sign",
    response_model=ClassificationResult
    )
def classify_sign(
    request: Request,
    file_img: UploadFile,
    model_name: str = 'cnn',
    redis_conn: redis.Redis = Depends(get_redis)
        ) -> ClassificationResult:
    """Predict the class of the sign in the image.

    Args:
        request (Request): request instance.
        file_img (UploadFile): uploaded image file.
        model_name (str, optional): model to use for prediction. Defaults to 'cnn'.
        redis_conn (redis.Redis, optional): redis connection. Defaults to Depends(get_redis).

    Returns:
        ClassificationResult: classification result with the detailed information.
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
        id=0,
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


@app.post("/detect_and_classify_signs")
async def detect_and_classify_signs(
    request: Request,
    file_data: UploadFile,
    classification_model: str = "cnn",
    detection_model: str = "yolo",
    suffix: str = "_",
    user_id: str = "0",
    redis_conn: redis.Redis = Depends(get_redis),
    postgres_session: AsyncSession = Depends(get_async_session)
        ) -> List[ClassificationResult]:
    """Detect and classify sign in the uploaded file. It could be image or video.

    Args:
        request (Request): Request instance.
        file_data (UploadFile): Uploaded image or video file.
        classification_model (str, optional): Model to use for classification. Defaults to "cnn".
        detection_model (str, optional): Model to use for prediction. Defaults to "yolo".
        suffix (str, optional): Suffix of the file (e.g. ".avi" or ".jpg").
        Uploaded file's suffix will be used if not provided. Defaults to "_".
        user_id (str, optional): User id. Will be randomly generated if not provided. Defaults to "0".
        redis_conn (redis.Redis, optional): Redis connection. Defaults to Depends(get_redis).
        postgres_session (AsyncSession, optional): Postgres connection. Defaults to Depends(get_async_session).

    Returns:
        List[ClassificationResult]: List with of all classified detections with the detailed information.
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

    data = SignDetection(tmp_path, user_id, MODELS.get_model(detection_model))
    data.detect()
    classification_results = []

    # # Generate cropped_signs from detected objects
    for frame_number, id, obj in data.stream_objects():
        sign = CroppedSign(
            user_id=user_id,
            result_filepath=data.annotated_filepath,
            detection_id=id,
            frame_number=frame_number,
            **obj
        )
        # Classify sign using cnn
        sign.classify(classification_model, MODELS.get_model(classification_model))
        logging.info(f"Prediction results: {sign.classification_results['cnn']}")
        res = ClassificationResult(**sign.to_postgres(classification_model))
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

    if classification_results:
        await write_results(classification_results, postgres_session)

    # Delete tmp file which we saved in the beggining
    tmp_path.unlink()

    return classification_results


@app.post("/map", response_class=HTMLResponse)
async def map(
    request: Request,
    file_data: UploadFile,
    suffix: str = "_",
    user_id: str = "0",
    redis_conn: redis.Redis = Depends(get_redis)
        ) -> folium.Map:
    """THIS IS A DEMO FUNCTION OF THE WORK IN PROGRESS FEATURE.
    Draw a map with the classification results.

    Args:
        request (Request): Request instance.
        file_data (UploadFile): Uploaded image or video file.
        suffix (str, optional): Suffix of the file (e.g. ".avi" or ".jpg").
        Uploaded file's suffix will be used if not provided. Defaults to "_".
        user_id (str, optional): User id. Will be randomly generated if not provided. Defaults to "0".
        redis_conn (redis.Redis, optional): Redis connection. Defaults to Depends(get_redis).

    Returns:
        folium.Map: HTML with all the classified detections rendered with the detailed information.
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

    # Create map object
    track = pd.read_csv("app/track.csv")[["lat", "lon"]].values.tolist()
    mapit = folium.Map(location=[55.7522, 37.6156], zoom_start=10)

    # Generate cropped_signs from detected objects
    for frame_number, id, obj in data.stream_objects():
        sign = CroppedSign(
            user_id=user_id,
            result_filepath=data.annotated_filepath,
            img=obj["img"],
            bbox=obj["bbox"],
            detection_id=id,
            detection_conf=obj["detection_conf"],
            frame_number=frame_number,
            detection_speed=obj["detection_speed"]
        )
        # Classify sign using cnn
        sign.classify_cnn(MODELS.cnn_model)
        logging.info(f"Prediction results: {sign.classification_results['cnn']}")

        res = ClassificationResult(**sign.to_postgres("cnn"))

        # Add Image
        encoded = base64.b64encode(sign.img)
        html = '<img src="data:image/png;base64,{}">'.format
        html = html(encoded.decode('UTF-8'))

        # Add table
        df_res = pd.DataFrame(res)
        html_df = df_res.T.to_html(
            classes="table table-striped table-hover table-condensed table-responsive")

        # Concatanate image and table
        html += html_df
        popup = folium.Popup(html, max_width="100%")

        # Add marker
        folium.Marker(
            location=[track[sign.frame_number // 30][0], track[sign.frame_number // 30][1]],
            tooltip="Нажмите для подробной информации!",
            popup=popup,
            icon=folium.Icon(color="blue"),
            radius=8
        ).add_to(mapit)

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

    # Return HTML with map
    return mapit.get_root().render()

# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
