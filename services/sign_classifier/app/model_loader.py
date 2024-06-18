import logging
import pathlib
from typing import Tuple

import joblib
import torch
from ultralytics import YOLO

from . import settings
from .cnn_model import gtrsb_model


class ModelLoader():
    def __init__(
            self,
            hog_model_path: pathlib.Path | str = settings.hog_model_path,
            kmeans_model_path: pathlib.Path | str = settings.kmeans_model_path,
            sift_model_path: pathlib.Path | str = settings.sift_model_path,
            cnn_model_path: pathlib.Path | str = settings.cnn_model_path,
            yolo_model_path: pathlib.Path | str = settings.yolo_model_path,
            cnn_dims: Tuple[int] = (7500, 106),
            ):
        # Load ML models
        # logging.info("Initialized ModelLoader instance.")
        # logging.info("Loading ML models...")
        # try:
        #     self.hog = joblib.load(hog_model_path)
        #     self.sift = {
        #         "kmeans": joblib.load(kmeans_model_path),
        #         "sift": joblib.load(sift_model_path)
        #     }
        #     logging.info("Successfully loaded ML models!")
        # except FileNotFoundError as fnfe:
        #     logging.error(f"Error occured while loading ML models:\n\n{fnfe}")

        # Init and load cnn_model
        logging.info("Loading DL models...")
        self.cnn = gtrsb_model(*cnn_dims)
        # self.cnn_model = joblib.load(cnn_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
            self.yolo = YOLO(yolo_model_path).to(self.device)
            logging.info("Successfully loaded DL models!")
        except FileNotFoundError as fnfe:
            logging.error(f"Error occured while loading DL models:\n\n {fnfe}")

    def get_model(self, model: str):
        return getattr(self, model, None)
