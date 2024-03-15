import base64
import logging
from datetime import datetime
from typing import Dict, List, Union

import redis
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.templating import Jinja2Templates

from . import classify

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

REDIS_CLIENT = redis.Redis(host='redis', port='6379', db=0)

# TODO: make this class more general, representing prediction instance for every endpoint/fucntion
class HistoryEntry():
    def __init__(self, image: bytes, model_used: str, success: int, sign_class: int, sign_description: str):
        self.image = image.decode('utf-8')
        self.model_used = model_used.decode('utf-8')
        self.success_status = success.decode('utf-8')
        self.sign_class = sign_class.decode('utf-8')
        self.sign_description = sign_description.decode('utf-8')


def append_history(image: bytes, model: str, data: Dict[str, Union[int, str, bool]]) -> bool:
    try:
        REDIS_CLIENT.xadd(
            "predictions:history",
            {
                "image": image,
                "model_used": model,
                "success": int(data["success"]),
                "sign_class": data.get("sign_class"),
                "sign_description": data.get("sign_description"),
            }
        )
        logging.info(f"Successfully added entry to the 'prediction:history'.")
        logging.info(f"Stream length: {REDIS_CLIENT.xlen('predictions:history')}")
        return True
    except Exception as e:
        logging.error(f"An error occured during redis transaction: {e}")
        return False


@app.get("/history")
def get_history(n_entries: int = 10):
    try:
        query = REDIS_CLIENT.xrange("predictions:history", "-", "+", n_entries)
        logging.debug(f"Query received: {query}")
        logging.info(f"Successfully received history of {len(query)} items")
    except Exception as e:
        logging.info(f"An error occured during redis transaction: {e}")

    # Make proper dict from query
    data = {}
    for item in query:
        # timestamp in seconds of an entry
        ts = int(item[0].decode('utf-8')[:10])
        hr_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S, %f')
        # cant decode image, not sure why, so i skip it for now
        data[hr_date] = {key.decode('utf-8'): value.decode('utf-8') for key, value in item[1].items() if key != b'image'}

    return data


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
        # timestamp in seconds of an entry
        ts = int(item[0].decode('utf-8')[:10])
        hr_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S, %f')
        # cant decode image, not sure why, so i skip it for now
        data = {key.decode('utf-8'): value for key, value in item[1].items()}
        history_list.append(HistoryEntry(**data))

    return templates.TemplateResponse("history.html", {"request": request, "history_list": history_list})

@app.post("/predict/sign_cnn")
def predict_one_sign_cnn(file: bytes = File(...)) -> Dict[str, Union[int, str, bool]]:
    """
    Predict traffic sign class with CNN model (one image).

    Args:
        - file (bytes): image represented by bytes.

    Returns:
        - data (dict): dict with info about image processing and sign class.
    """
    data = classify.predict_cnn_image(file)

    append_history(base64.b64encode(file).decode("utf-8"), model='cnn', data=data)

    return data


@app.post("/predict/signs_cnn")
def predict_many_signs_cnn(files: List[UploadFile] = File(...)) \
        -> Dict[str, Dict[str, Union[int, str, bool]]]:
    """
    Predict traffic sign class with CNN model (many images).

    Args:
        - files (list): list with images represented by bytes.

    Returns:
        - data (dict): dict with info about images processing and signs class.
    """
    data = {}
    image_list = []

    for file in files:
        try:
            image_list.append(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    im_num = 0
    for image in image_list:
        im_num += 1
        data.update({str(im_num) + " image": classify.predict_cnn_image(image)})

    return data


@app.post("/predict/sign_hog")
def predict_one_sign_hog(file: bytes = File(...)) -> Dict[str, Union[int, str, float, bool]]:
    """
    Predict traffic sign class on the image HOG features extraction (one image).

    Args:
        - file (bytes): image represented by bytes.

    Returns:
        - data (dict): dict with info about image processing and sign class.
    """
    data = classify.predict_hog_image(file)

    return data


@app.post("/predict/signs_hog")
def predict_many_signs_hog(files: List[UploadFile] = File(...)) \
        -> Dict[str, Dict[str, Union[int, str, float, bool]]]:
    """
    Predict traffic sign class on the image HOG features extraction (many images).

    Args:
        - files (list): list with images represented by bytes.

    Returns:
        - data (dict): dict with info about images processing and signs class.
    """
    data = {}
    image_list = []

    for file in files:
        try:
            image_list.append(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    im_num = 0
    for image in image_list:
        im_num += 1
        data.update({str(im_num) + " image": classify.predict_hog_image(image)})

    return data


@app.post("/predict/sign_sift")
def predict_one_sign_sift(file: bytes = File(...)) -> Dict[str, Union[int, str, float, bool]]:
    """
    Predict traffic sign class on the image using SIFT features extraction (one image).

    Args:
        - file (bytes): image represented by bytes.

    Returns:
        - data (dict): dict with info about image processing and sign class.
    """
    data = classify.predict_sift_image(file)

    return data


@app.post("/predict/signs_sift")
def predict_many_signs_sift(files: List[UploadFile] = File(...)) \
        -> Dict[str, Dict[str, Union[int, str, float, bool]]]:
    """
    Predict traffic sign class on the image SIFT features extraction (many images).

    Args:
        - files (list): list with images represented by bytes.

    Returns:
        - data (dict): dict with info about images processing and signs class.
    """
    data = {}
    image_list = []

    for file in files:
        try:
            image_list.append(file.file.read())
        except Exception:
            return {"message": "There was an error uploading the file(s)"}
        finally:
            file.file.close()

    im_num = 0
    for image in image_list:
        im_num += 1
        data.update({str(im_num) + " image": classify.predict_sift_image(image)})

    return data


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
