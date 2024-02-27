import io
import logging
from typing import Dict, Union

import cv2
import joblib
import numpy as np
from skimage.feature import hog

import PIL
import torch
#import torch.nn as nn
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize, ToTensor


from . import settings
from .cnn_model import GTSRB_MODEL

# first iteration models
HOG_MODEL = joblib.load(settings.hog_model_path)
KMEANS_MODEL = joblib.load(settings.kmeans_model_path)
SIFT_MODEL = joblib.load(settings.sift_model_path)

# second iteration model
EPOCHS = 20
LEARNING_RATE = 0.0008
INPUT_DIM = 3*50*50
OUTPUT_DIM = 43
CNN_MODEL = GTSRB_MODEL(INPUT_DIM, OUTPUT_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CNN_MODEL.load_state_dict(torch.load(settings.cnn_model_path, map_location=device))

SIGNS_DESC = {
    0: 'Speed Limit 20 kmph',
    1: 'Speed Limit 30 kmph',
    2: 'Speed Limit 50 kmph',
    3: 'Speed Limit 60 kmph',
    4: 'Speed Limit 70 kmph',
    5: 'Speed Limit 80 kmph',
    6: 'End of Speed Limit 80 kmph',
    7: 'Speed Limit 100 kmph',
    8: 'Speed Limit 120 kmph',
    9: 'No Passing',
    10: 'No Passing vehicle over 3,5 ton',
    11: 'Right of way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh over 3,5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice or snow',
    31: 'Wild animals crossing',
    32: 'End speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle over 3,5 tons'
}


def read_cv2_image(binaryimg: bytes) -> np.array:
    """
    Read passed image to numpy array.

    Args:
        - binaryimg (bytes): bytes representing an image.

    Returns:
        - image (np.array): np.array representing an image.
    """
    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    #image = cv2.imdecode(image, cv2.IMREAD_ANYCOLOR)
    image = PIL.Image.open(io.BytesIO(image)).convert('RGB')
    #logging.debug(f'Image resolution: {image.shape}.')

    return np.array(image)


def predict_hog_image(binaryimg: bytes) -> Dict[str, Union[int, str, float, bool]]:
    """
    Load and preprocess image containing a traffic sign, predict what that sign
    using HOG feature extraction.

    Args:
        - binaryimg (bytes): bytes representing an image.

    Returns:
        - data (dict): dict with info about image processing.
    """
    data = {"success": False}
    if binaryimg is None:
        return data

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    # convert the image to grayscale
    im_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)]).astype(float)

    # computes HOG features of input images
    hog_output = [hog(img, pixels_per_cell=(2, 2), visualize=True) for img in im_gray]
    data_hog = [hog_img for out, hog_img in hog_output]

    # resize the image
    size = (50, 50)
    im_resized = np.array([cv2.resize(img, size) for img in data_hog], dtype=object).astype(float)

    # image flattening, reshaping the data to the (samples, feature) matrix format
    data_reshaped = im_resized.reshape((len(im_resized), -1))

    # prediction
    prediction = HOG_MODEL['best_model'].predict(data_reshaped)
    pred_class = int(prediction.tolist()[0])
    confidence = HOG_MODEL['best_model'].predict_proba(data_reshaped)[0][pred_class-1]

    data.update(
        {
            "sign_class": pred_class,
            "sign_description": SIGNS_DESC[pred_class],
            "confidence": float(f"{confidence.tolist():.4f}"),
            "success": True
        }
    )

    return data


def preprocess_sift_image(img: np.array, img_shape: tuple = (32, 32)) -> np.array:
    """
    Preprocess image, represented by np.array.

    Args:
        - img (np.array): image represented by np.array.
        - img_shape (typle=(32, 32)): resulting shape of an image, defaults to 32x32.

    Returns:
        - normalized_gray_img (np.array): preprocessed image represented as np.array.
    """
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
        normalized_gray_img = cv2.resize(normalized_gray_img, img_shape, interpolation=cv2.INTER_AREA)
    else:
        normalized_gray_img = cv2.resize(normalized_gray_img, img_shape, interpolation=cv2.INTER_CUBIC)

    return normalized_gray_img


def predict_sift_image(binaryimg: bytes) -> Dict[str, Union[int, str, float, bool]]:
    """
    Load and preprocess image containing a traffic sign, predict what that sign
    using SIFT feature extraction.

    Args:
        - binaryimg (bytes): bytes representing an image.

    Returns:
        - data (dict): dict with info about image processing.
    """
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
    except ValueError:
        # Skipping when no descriptors are found
        pass

    for idx in closest_idx:
        features[idx] += 1

    logging.debug(f'SIFT feature vector: {features}')

    pred_class = int(SIFT_MODEL.predict(features.reshape(1, -1))[0])

    # We cant predict confidence level in that case, so for the uniformity we say that its 100.0.
    data.update(
        {
            "sign_class": pred_class,
            "sign_description": SIGNS_DESC[pred_class],
            "confidence": 100.0,
            "success": True
        }
    )

    return data


def predict_cnn_image(binaryimg: bytes):
    """
    Load and preprocess image containing a traffic sign, predict what that sign
    using CNN model.

    Args:
        - binaryimg (bytes): bytes representing an image.

    Returns:
        - data (dict): dict with info about image processing.
    """
    data = {"success": False}
    if binaryimg is None:
        return data

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = CNN_MODEL.eval().to(device)

    model = CNN_MODEL.eval()

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    # convert to tensor, normalize,
    # swap axes to [channels, H, W] format
    image = torch.from_numpy(image) / 255
    image = torch.swapaxes(image, 0, 2)
    image = torch.swapaxes(image, 1, 2)


    transforms = Compose(
        [
            Resize([50, 50]),
            ToTensor()
        ]
    )

    image = transforms(image)

    # prediction
    with torch.no_grad() :
        prediction = model.forward(image[None, :, :, :])
        _, pred = torch.max(prediction, 1)

    data.update(
            {
                "sign_class": pred.item(),
                "sign_description": SIGNS_DESC[pred.item()],
                "success": True
            }
        )

    return data