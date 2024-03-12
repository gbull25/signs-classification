import logging
from typing import Dict, List, Union

import uvicorn
from fastapi import FastAPI, File, UploadFile

from . import classify
from fastapi import FastAPI, Depends
from fastapi_users import FastAPIUsers
from postgre.auth.auth import auth_backend
from postgre.auth.database import User
from postgre.auth.manager import get_user_manager
from postgre.auth.schemas import UserRead, UserCreate

app = FastAPI(
    # lifespan=lifespan,
    title="Predict the class of a sign",
    version="1.0",
    contact={
        "name": "Gleb Bulygin & Viktor Tikhomirov",
        "url": "https://github.com/gbull25/signs-classification"
    }
    )

# @app.get()

fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

# Добавляем возможность регистрации
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

current_user = fastapi_users.current_user()


@app.get("/protected-route")
def protected_route(user: User = Depends(current_user)):
    return f"Hello, {user.username}"


@app.get("/unprotected-route")
def unprotected_route():
    return f"Hello, anonym"


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
