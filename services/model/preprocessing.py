import io
import lzma
import pickle

import cv2
import joblib
import numpy as np
from skimage.feature import hog

HOG_MODEL = joblib.load("services/model/lzma_hog_proba.xz")
KMEANS_MODEL = joblib.load("services/model/kmeans.gz")
SIFT_MODEL = joblib.load("services/model/sift_svm.gz")


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
    prediction = HOG_MODEL['best_model'].predict(data_reshaped)
    pred_class = int(prediction.tolist()[0])
    confidence = HOG_MODEL['best_model'].predict_proba(data_reshaped)[0][pred_class-1]


    data.update({"sign class": pred_class, 
                 "confidence": float(f"{confidence.tolist():.4f}"), 
                 "success": True})

    return data


def preprocess_sift_image(img: np.array, img_shape: tuple = (32, 32)) -> np.array:

    # Equalize histogram, convert to gray
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    bgr_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    # Normalize image
    normalized_gray_img = np.zeros(gray_img.shape)
    normalized_gray_img = cv2.normalize(gray_img,  normalized_gray_img, 0, 255, cv2.NORM_MINMAX)

    # Resize image to img_shape. Use different interpolation whether we shrink or enlarging the image.
    if normalized_gray_img.size >= np.prod(np.array(img_shape)):
        normalized_gray_img = cv2.resize(normalized_gray_img, img_shape, interpolation = cv2.INTER_AREA)
    else:
        normalized_gray_img = cv2.resize(normalized_gray_img, img_shape, interpolation = cv2.INTER_CUBIC)

    return normalized_gray_img


def predict_sift_image(binaryimg):
     
    data = {"success": False}
    if binaryimg is None:
        return data
    
    # Initialize feature vector
    features = np.zeros(645)
    # Read image
    image = read_cv2_image(binaryimg)
    # Preprocess image
    prep_image = preprocess_sift_image(image)

    sift = cv2.SIFT_create(sigma=1.0)
    _, des = sift.detectAndCompute(prep_image, None)

    try:
        closest_idx = KMEANS_MODEL.predict(des)
    except ValueError as ve:
        # Skipping when des is empty
        pass

    print('im here')
    for idx in closest_idx:
        features[idx] += 1

    pred_cls = SIFT_MODEL.predict(features.reshape(1, -1))

    data.update(
        {
            "sign class": int(pred_cls[0]), 
            "confidence": 100.0, 
            "success": True
        }
    )

    return data
