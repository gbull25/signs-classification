import base64
import io
import logging
from copy import deepcopy
from typing import Dict, Union

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import hog
from torchvision.transforms.v2 import Compose, Resize, ToTensor
    

class YOLO_detect():
    def __init__(self):
        pass
    
    def process_photo(self, video_name, yolo_model):
        """
        
        """
        detections = yolo_model(video_name, save=True, project='./video', conf=0.51)

        return 'Success'

    def process_video(self, video_name, yolo_model):
        """
        
        """

        # Read image
        #local_img = np.array(Image.open(io.BytesIO(self.img)).convert('RGB'))

        # Predict 
        detections = yolo_model.track(video_name, stream=True, save=True, name='', project='./video', conf=0.51)

        self.yolo_detection_result = []
        bbox_lst = []
        crop_lst = []
        id_lst = []

        # Read result object for every image
        for detect in detections:
            self.yolo_detection_result.append(detect)
            # Image name
            #img_name = detect.path.split('/')[-1]

            # Read bboxes  
            if detect.boxes.id != None:
                idx = detect.boxes.id.tolist()
                for index, sign_id in enumerate(idx):
                    if sign_id not in id_lst:
                        id_lst.append(sign_id)

                        sign_bbox = detect.boxes.xyxy.tolist()[index]
                        x1, y1, x2, y2 = map(int, sign_bbox)

                        # Crop image
                        crop_img = detect.orig_img[y1:y2, x1:x2]

                        # Save results
                        bbox_lst.append(sign_bbox)
                        crop_lst.append(crop_img)

            # Return dict for making response
        id_lst.sort()
        return 'Success'
        #return {
        #    "signs_bboxes": bbox_lst,
        #    "crop_image": crop_lst,
        #    "unique_ids": id_lst
        #}