import os
import pathlib

kmeans_model_path = os.getenv('KMEANS_MODEL_PATH', pathlib.Path('/models/kmeans.gz'))
sift_model_path = os.getenv('SIFT_MODEL_PATH', pathlib.Path('/models/sift_svm.gz'))
hog_model_path = os.getenv('HOG_MODEL_PATH', pathlib.Path('/models/lzma_hog_proba.xz'))
cnn_model_path = os.getenv('CNN_MODEL_PATH', pathlib.Path('/models/cnn_torch.pt'))
