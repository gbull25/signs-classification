import base64
import csv
import io
import logging
import pathlib
import time
from copy import deepcopy
from typing import Dict, Union

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
from torchvision.transforms.v2 import Compose, Resize, ToTensor


class CroppedSign():

    # describe_by_class = {
    #     0: 'Speed Limit 20 kmph',
    #     1: 'Speed Limit 30 kmph',
    #     2: 'Speed Limit 50 kmph',
    #     3: 'Speed Limit 60 kmph',
    #     4: 'Speed Limit 70 kmph',
    #     5: 'Speed Limit 80 kmph',
    #     6: 'End of Speed Limit 80 kmph',
    #     7: 'Speed Limit 100 kmph',
    #     8: 'Speed Limit 120 kmph',
    #     9: 'No Passing',
    #     10: 'No Passing vehicle over 3,5 ton',
    #     11: 'Right of way at intersection',
    #     12: 'Priority road',
    #     13: 'Yield',
    #     14: 'Stop',
    #     15: 'No vehicles',
    #     16: 'Veh over 3,5 tons prohibited',
    #     17: 'No entry',
    #     18: 'General caution',
    #     19: 'Dangerous curve left',
    #     20: 'Dangerous curve right',
    #     21: 'Double curve',
    #     22: 'Bumpy road',
    #     23: 'Slippery road',
    #     24: 'Road narrows on the right',
    #     25: 'Road work',
    #     26: 'Traffic signals',
    #     27: 'Pedestrians',
    #     28: 'Children crossing',
    #     29: 'Bicycles crossing',
    #     30: 'Beware of ice or snow',
    #     31: 'Wild animals crossing',
    #     32: 'End speed and passing limits',
    #     33: 'Turn right ahead',
    #     34: 'Turn left ahead',
    #     35: 'Ahead only',
    #     36: 'Go straight or right',
    #     37: 'Go straight or left',
    #     38: 'Keep right',
    #     39: 'Keep left',
    #     40: 'Roundabout mandatory',
    #     41: 'End of no passing',
    #     42: 'End no passing vehicle over 3,5 tons'
    # }
    describe_by_class = {}

    with open("app/numbers_to_classes.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            describe_by_class[int(row[0])] = row[1]

    def __init__(
            self,
            user_id: int,
            source_filepath: str | pathlib.Path,
            img: bytes,
            bbox: str | None = None,
            id: int | None = None,
            frame_number: int = 1,
            detection_speed: float = 0,
            classification_speed: float = 0,
            hog_result_class: int | None = None,
            hog_result_description: str | None = None,
            sift_result_class: int | None = None,
            sift_result_description: str = None,
            cnn_result_class: int | None = None,
            cnn_result_description: str | None = None,
            ):

        self.user_id = user_id
        if isinstance(source_filepath, pathlib.Path):
            self.source_filepath = str(source_filepath)
        else:
            self.source_filepath = source_filepath
        self.img = img
        self.bbox = bbox
        self.id = id
        self.frame_number = frame_number
        self.detection_speed = detection_speed
        self.classification_speed = classification_speed


        self.hog_result_class = hog_result_class
        self.hog_result_description = hog_result_description
        self.sift_result_class = sift_result_class
        self.sift_result_description = sift_result_description
        self.cnn_result_class = cnn_result_class
        self.cnn_result_description = cnn_result_description

    def timer(func):
        def wrapper(self, *arg, **kw):
            t1 = time.time()
            res = func(self, *arg, **kw)
            t2 = time.time()
            self.classification_speed = (t2 - t1)
            return res
        return wrapper

    def preprocess_for_sift(self, img_shape=(32, 32)):
        """
        Preprocess image, represented by np.array.

        Store preprocessed image in the new attribute "preprocessed_sift_img".

        Args:
            - img_shape (tuple=(32, 32)): resulting shape of an image, defaults to 32x32.
        """
        # Equalize histogram, convert to gray
        img = np.array(Image.open(io.BytesIO(self.img)).convert('RGB'))
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

        # Save preprocessed image
        self.preprocessed_sift_img = normalized_gray_img

    def classify_hog(self, svc_hog_model):
        """
        Classify cropped sign image with SVC model based on HOG feature extraction.
        """
        # Convert the image to grayscale
        colored_img = np.array(Image.open(io.BytesIO(self.img)).convert('RGB'))
        im_gray = np.array([cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)]).astype(float)

        # Compute HOG features of input images
        hog_output = [hog(img, pixels_per_cell=(2, 2), visualize=True) for img in im_gray]
        data_hog = [hog_img for out, hog_img in hog_output]

        # Resize the image
        size = (50, 50)
        im_resized = np.array([cv2.resize(img, size) for img in data_hog], dtype=object).astype(float)

        # Image flattening, reshaping the data to the (samples, feature) matrix format
        data_reshaped = im_resized.reshape((len(im_resized), -1))

        # Predict
        prediction = svc_hog_model['best_model'].predict(data_reshaped)
        pred_class = int(prediction.tolist()[0])

        # Save results
        self.hog_result_class = pred_class
        self.hog_result_description = self.describe_by_class[pred_class]

        # Return dict for making response
        return {
            "sign_class": self.hog_result_class,
            "sign_description": self.hog_result_description,
            "model_used": "cnn_model"
        }

    def classify_sift(self, kmeans_model, svc_sift_model):
        """
        Classify cropped sign image with SVC model based on SIFT feature extraction.
        """
        # Initialize feature vector
        features = np.zeros(645)
        # Preprocess image
        self.preprocess_for_sift()

        sift = cv2.SIFT_create(sigma=1.0)
        _, des = sift.detectAndCompute(self.preprocessed_sift_img, None)

        try:
            closest_idx = kmeans_model.predict(des)
        except ValueError:
            # Skip when no descriptors are found
            pass

        for idx in closest_idx:
            features[idx] += 1

        logging.debug(f'SIFT feature vector: {features}')

        pred_class = int(svc_sift_model.predict(features.reshape(1, -1))[0])

        # Save results
        self.sift_result_class = pred_class
        self.sift_result_description = self.describe_by_class[pred_class]

        # Return dict for making response
        return {
            "sign_class": self.sift_result_class,
            "sign_description": self.sift_result_description,
            "model_used": "cnn_model"
        }

    @timer
    def classify_cnn(self, cnn_model):
        """
        Classify cropped sign image with CNN model.
        """
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = CNN_MODEL.eval().to(device)

        model = cnn_model.eval()

        # Convert to tensor, normalize,
        # Swap axes to [channels, H, W] format
        local_img = np.array(Image.open(io.BytesIO(self.img)).convert('RGB'))
        image = torch.from_numpy(local_img) / 255
        image = torch.swapaxes(image, 0, 2)
        image = torch.swapaxes(image, 1, 2)

        transforms = Compose(
            [
                Resize([50, 50]),
                ToTensor()
            ]
        )

        image = transforms(image)

        # Predict
        with torch.no_grad():
            prediction = model.forward(image[None, :, :, :])
            _, pred_class = torch.max(prediction, 1)

        # Save results
        self.cnn_result_class = pred_class.item()
        self.cnn_result_description = self.describe_by_class[pred_class.item()]

        # Return dict for making response
        return {
            "user_id": self.user_id,
            "result_filepath": self.source_filepath,
            "detection_id": self.id,
            "detection_conf": 0,
            "sign_class": int(self.cnn_result_class),
            "sign_description": self.cnn_result_description,
            "bbox": self.bbox,
            "frame_number": self.frame_number,
            "detection_speed": self.detection_speed,
            "model_used": "cnn_model"
        }

    def to_redis(self) -> Dict[str, Union[str, int]]:
        """
        Make modified dict of the self attributes to store in redis.
        """
        res = {}

        for key, val in self.__dict__.items():
            logging.error(f"FORMING REDIS MSG:")
            logging.error(f"{key}: {val}")
            if key == "img":
                res[key] = val
                continue
            if val is None:
                res[key] = "no_data"
            else:
                res[key] = val
            logging.info(f"Key: {key}, {type(key)}, Value: {val}, {type(val)}")

        return res

    def to_html(self) -> Dict[str, Union[str, int]]:
        """"""
        res = deepcopy(self.__dict__)
        res["img"] = base64.b64encode(res["img"]).decode("utf-8")

        return res

    @classmethod
    def from_redis(cls, init_data):
        """Call init with data from redis"""
        res = {}
        for key, val in init_data.items():
            key = key.decode('utf-8')
            if key == 'img':
                res[key] = val
                continue
            val = val.decode('utf-8')
            logging.info(f"Key: {key}, {type(key)}, Value: {val}, {type(val)}")
            if val == "no_data":
                continue
            if val.isdigit():
                res[key] = int(val)
            else:
                res[key] = val

        return cls(**res)
