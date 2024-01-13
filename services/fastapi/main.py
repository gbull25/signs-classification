from typing import List

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile

from services.model import preprocessing

app = FastAPI(
    # lifespan=lifespan,
    title="Predict the class of a sign",
    version="1.0",
    contact={
        "name": "Gleb Bulygin & Viktor Tikhomirov",
        "url": "https://github.com/gbull25/signs-classification"
    }
    )


@app.post("/predict/sign_hog")
def predict_one_sign_hog(request: Request, file: bytes = File(...)):

    data = {"success": False}
    data = preprocessing.predict_hog_image(file)

    return data


@app.post("/predict/signs_hog")
def predict_many_signs_hog(request: Request, files: List[UploadFile] = File(...)):

    data = {"success": False}
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
        data.update({str(im_num)+" image":preprocessing.predict_hog_image(image)})

    data.update({"success": True})

    return data


@app.post("/predict/sign_sift")
def predict_one_sign_sift(request: Request, file: bytes = File(...)):

    data = {"success": False}
    data = preprocessing.predict_sift_image(file)

    return data


@app.post("/predict/signs_sift")
def predict_many_signs_sift(request: Request, files: List[UploadFile] = File(...)):

    data = {"success": False}
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
        data.update({str(im_num)+" image":preprocessing.predict_sift_image(image)})

    data.update({"success": True})

    return data

if __name__ == "__main__":
    uvicorn.run(app, port=8000)