import numpy as np
import io
import pickle
import cv2
import lzma
from skimage.feature import hog

# Загрузка сжатых моделей для телеграм бота
with lzma.open("services/model/lzma_hog_proba.xz", "rb") as f:
        model_data = pickle.load(f)

# Загрузка сжатых моделей для сервиса Fastapi
def load_model_data():
    with lzma.open("services/model/lzma_hog_proba.xz", "rb") as f:
        model_data = pickle.load(f)

    return model_data

def read_cv2_image(binaryimg):

    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

def predict_hog_image(binaryimg):
    
    data = {"success": False}
    if binaryimg is None:
        return data

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    # convert the image to grayscale
    im_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]).astype(float)

    # computes HOG features of input images
    hog_output = [hog(img, pixels_per_cell = (2, 2), visualize = True) for img in im_gray]
    data_hog = [hog_img for out, hog_img in hog_output]

    # resize the image
    size = (50,50)
    im_resized = np.array([cv2.resize(img, size) for img in data_hog], dtype=object).astype(float)

    # image flattening, reshaping the data to the (samples, feature) matrix format
    data_reshaped = im_resized.reshape((len(im_resized), -1))

    # prediction
    prediction = model_data['best_model'].predict(data_reshaped)
    pred_class = int(prediction.tolist()[0])
    confidence = model_data['best_model'].predict_proba(data_reshaped)[0][pred_class-1]


    data.update({"sign class": pred_class, 
                 "confidence": float(f"{confidence.tolist():.4f}"), 
                 "success": True})

    return data