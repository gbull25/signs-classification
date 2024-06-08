import os
import pathlib

kmeans_model_path = os.getenv('KMEANS_MODEL_PATH', pathlib.Path('/models/kmeans.gz'))
sift_model_path = os.getenv('SIFT_MODEL_PATH', pathlib.Path('/models/sift_svm.gz'))
hog_model_path = os.getenv('HOG_MODEL_PATH', pathlib.Path('/models/lzma_hog_proba.xz'))
cnn_model_path = os.getenv('CNN_MODEL_PATH', pathlib.Path('/models/cnn_rtsd_final.pt'))
yolo_model_path = os.getenv('YOLO_MODEL_PATH', pathlib.Path('/models/yolo_best_10epochs.pt'))

user_data_storage_path = os.getenv('USER_DATA_STORAGE_PATH', pathlib.Path('/user_data'))