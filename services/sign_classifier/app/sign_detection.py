import pathlib
import time
from collections import defaultdict
from typing import Dict, Union

import cv2

from . import settings


class SignDetection():

    def __init__(self, data: pathlib.Path | str, user_id: str, yolo_model) -> None:
        self.data = data
        self.project_path = settings.user_data_storage_path / user_id / str(int(round(time.time() * 1000)))
        self.project_path.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pathlib.Path):
            self.annotated_filepath = self.project_path / "track" / self.data.name
        else:
            self.annotated_filepath = self.project_path / "track" / "youtube_link.avi"
        self.yolo_model = yolo_model
        self.detection_result = []
        self.objects: Dict[int, Dict[str, Dict[str, Union[str, bytes, float]]]] = defaultdict(dict)

    def detect(self):
        detections_stream = self.yolo_model.track(
            self.data,
            conf=0.5,
            stream=True,
            save=True,
            project=self.project_path
            )
        for frame_number, detection in enumerate(detections_stream):
            if not detection:
                # Add msg that no signs was detected
                continue
            self.detection_result.append(detection)
            self._process_detection(frame_number+1, detection)

    def _process_detection(self, frame_number, detection):
        orig_image = detection.orig_img
        for obj in detection:
            # Somehow there can be no id (id=None)
            if obj.boxes.id:
                id = int(obj.boxes.id.item())
            else:
                continue
            x1, y1, x2, y2 = map(int, obj.boxes.xyxy.tolist()[0])
            cropped_img = orig_image[y1:y2, x1:x2]
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
            _, cropped_img_bytes = cv2.imencode('.jpg', cropped_img, encode_params)
            self.objects[frame_number][id] = {
                    "bbox": str([x1, y1, x2, y2]),
                    "img": cropped_img_bytes.tobytes(),
                    "detection_conf": obj.boxes.conf.item(),
                    "detection_speed": sum(detection.speed.values())
                }

    def stream_objects(self):
        for frame_number, id in self.objects.items():
            for id, obj in id.items():
                yield frame_number, id, obj
