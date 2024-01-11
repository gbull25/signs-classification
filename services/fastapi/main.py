from fastapi import FastAPI, File, UploadFile, Request
from typing import List
import uvicorn

from services.model import preprocessing

app = FastAPI()


@app.post("/predict/sign")
def predict_one_sign(request: Request, file: bytes = File(...)):

    data = {"success": False}
    data = preprocessing.predict_hog_image(file)

    return data


@app.post("/predict/signs")
def predict_many_signs(request: Request, files: List[UploadFile] = File(...)):

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

if __name__ == "__main__":
    uvicorn.run(app, port=8000)