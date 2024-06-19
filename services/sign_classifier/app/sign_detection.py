import pathlib
import time
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple, Union

import cv2

from . import settings


class SignDetection():
    """Representation of the file, either a photo or a video.
    All the data and functions associated with it for detection process is attached.

    Attributes:
        data (pathlib.Path): Path to the file.
        user_id (str): Id of the user who uploaded this file.
        yolo_model (Any): YOLO detection model.
        project_path (pathlib.Path): Path to the folder where annotated files will be put.
        annotated_filepath (pathlib.Path): Path to the annotated file.
        detection_result (List[Any]): List with detection instances found in this file.
        objects (Dict[int, Dict[str, Dict[str, Union[str, bytes, float]]]]): Detected objects with all the information.
    """

    def __init__(self, data: pathlib.Path, user_id: str, yolo_model: Any) -> None:
        """Initialize class instance."""
        self.data = data
        self.project_path = settings.user_data_storage_path / user_id / str(int(round(time.time() * 1000)))
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.annotated_filepath = self.project_path / "track" / self.data.name
        self.yolo_model = yolo_model
        self.detection_result: List[Any] = []
        self.objects: Dict[int, Dict[str, Dict[str, Union[str, bytes, float]]]] = defaultdict(dict)

    def detect(self):
        """Run the YOLO model inference."""
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

    def _process_detection(self, frame_number: int, detection: Any):
        """Extract and process crucial information from detection instance.

        Args:
            frame_number (int): Number of a frame in which detection was observed.
            detection (Any): Detection instance.
        """
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

    def stream_objects(self) -> Iterator[Tuple[int, Dict[str, Union[str, bytes, float]]]]:
        """Stream found objects one by one.

        Yields:
            Iterator[Tuple[int, Dict[str, Union[str, bytes, float]]]]: frame number, detection id and dict with the
            crucial information about the detection.
        """
        for frame_number, id in self.objects.items():
            for id, obj in id.items():
                yield frame_number, id, obj
