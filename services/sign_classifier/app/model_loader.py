import logging
import pathlib
from typing import Tuple

import joblib
import torch

from . import settings
from .cnn_model import gtrsb_model


class ModelLoader():
    def __init__(
            self,
            hog_model_path: pathlib.Path | str = settings.hog_model_path,
            kmeans_model_path: pathlib.Path | str = settings.kmeans_model_path,
            sift_model_path: pathlib.Path | str = settings.sift_model_path,
            cnn_model_path: pathlib.Path | str = settings.cnn_model_path,
            cnn_dims: Tuple[int] = (7500, 43),
        ):
        # Load ML models
        logging.info(f"Initialized ModelLoader instance.")
        logging.info(f"Loading HOG model...")
        try:
            self.hog_model = joblib.load(hog_model_path)
            self.kmeans = joblib.load(kmeans_model_path)
            self.sift_model = joblib.load(sift_model_path)
            logging.info(f"Successfully loaded ML models!")
        except FileNotFoundError as fnfe:
            logging.error(f"Error occured while loading ML models:\n\n{fnfe}")

        # Init and load cnn_model
        logging.info(f"Loading DL models...")
        self.cnn_model = gtrsb_model(*cnn_dims)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
            logging.info(f"Successfully loaded DL models!")
        except FileNotFoundError as fnfe:
            logging.error(f"Error occured while loading DL models:\n\n {fnfe}")