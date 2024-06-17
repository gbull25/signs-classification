import base64
import csv
import io
import logging
import pathlib
import time
from collections import defaultdict
from typing import Dict, Union

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
from torchvision.transforms.v2 import Compose, Resize, ToTensor


class CroppedSign():

    describe_by_class = {}

    with open("app/numbers_to_classes.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            describe_by_class[int(row[0])] = row[1] + ':' + row[2]

    def __init__(
            self,
            user_id: int,
            result_filepath: str | pathlib.Path,
            img: bytes,
            bbox: str | None = None,
            detection_id: int | None = None,
            detection_conf: float = 0,
            frame_number: int = 1,
            detection_speed: float = 0,
            classification_speed: float = 0,
            classification_results: dict = {}
            ):

        self.user_id = user_id
        if isinstance(result_filepath, pathlib.Path):
            self.result_filepath = str(result_filepath)
        else:
            self.result_filepath = result_filepath
        self.img = img
        self.bbox = bbox
        self.detection_id = detection_id
        self.detection_conf = detection_conf
        self.frame_number = frame_number
        self.detection_speed = detection_speed
        self.classification_speed = classification_speed
        if not classification_results:
            self.classification_results: Dict[str, Dict[str, Union[str, float]]] = defaultdict(dict)
        else:
            self.classification_results = classification_results

    def timer(model_name):
        def inner(func):
            def wrapper(self, *arg, **kw):
                t1 = time.time()
                res = func(self, *arg, **kw)
                t2 = time.time()
                self.classification_results[model_name]["classification_speed"] = (t2 - t1)
                return res
            return wrapper
        return inner

    def _preprocess_for_sift(self, img_shape=(32, 32)):
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
        return normalized_gray_img

    @timer("hog")
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

        self.classification_results["hog"] = {
            "sign_class": pred_class,
            "sign_description": self.describe_by_class[pred_class]
        }

    @timer("sift")
    def classify_sift(self, sift_model):
        """
        Classify cropped sign image with SVC model based on SIFT feature extraction.
        """
        kmeans_model = sift_model["kmeans"]
        sift_model = sift_model["sift"]
        # Initialize feature vector
        features = np.zeros(645)
        # Preprocess image
        preprocessed_img = self._preprocess_for_sift()

        sift = cv2.SIFT_create(sigma=1.0)
        _, des = sift.detectAndCompute(preprocessed_img, None)

        try:
            closest_idx = kmeans_model.predict(des)
        except ValueError:
            # Skip when no descriptors are found
            pass

        for idx in closest_idx:
            features[idx] += 1

        logging.debug(f'SIFT feature vector: {features}')

        pred_class = int(sift_model.predict(features.reshape(1, -1))[0])

        self.classification_results["sift"] = {
            "sign_class": pred_class,
            "sign_description": self.describe_by_class[pred_class]
        }

    @timer("cnn")
    def classify_cnn(self, cnn_model):
        """
        Classify cropped sign image with CNN model.
        """
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

        # Return dict for making response
        self.classification_results["cnn"] = {
            "sign_class": pred_class.item(),
            "sign_description": self.describe_by_class[pred_class.item()]
        }

    def classify(self, model_name, classification_model):
        classification_method = getattr(self, f"classify_{model_name}")
        classification_method(classification_model)

    def to_redis(self) -> Dict[str, Union[str, int]]:
        """
        Make modified dict of the self attributes to store in redis.
        """
        res = dict(self)
        classification_results = res.pop("classification_results")
        for model, results in classification_results.items():
            for param, value in results.items():
                new_key = model + "-" + param
                res[new_key] = value

        return res

    def to_postgres(self, model_used):
        res = dict(self)
        # res = deepcopy(self.__dict__)
        _ = res.pop("img")
        classification_results = res.pop("classification_results")

        for key, val in classification_results[model_used].items():
            res[key] = val

        res["model_used"] = model_used

        return res

    def to_html(self) -> Dict[str, Union[str, int]]:
        """"""
        res = dict(self)
        res["img"] = base64.b64encode(res["img"]).decode("utf-8")

        return res

    @classmethod
    def from_redis(cls, init_data):
        """Call init with data from redis"""
        res = {}
        classification_models = ("cnn", "hog", "sift")
        classification_results = defaultdict(dict)
        for key, val in init_data.items():
            key = key.decode('utf-8')

            if key.startswith(classification_models):
                model, param = key.split("-")
                classification_results[model][param] = val.decode("utf-8")
                continue

            if key == 'img':
                res[key] = val
                continue

            val = val.decode('utf-8')
            if val == "no_data":
                continue
            if val.isdigit():
                res[key] = int(val)
            else:
                res[key] = val

            logging.info(f"Key: {key}, {type(key)}, Value: {val}, {type(val)}")

        res["classification_results"] = classification_results

        return cls(**res)

    def __iter__(self):
        for key in self.__dict__.keys():
            yield key, getattr(self, key)
