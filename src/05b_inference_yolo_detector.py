import os
import pathlib
import pickle
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from yolo_utils import get_center_of_bbox, measure_distance, read_video, save_video
from model_maps import CPIECE2IDX

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent
MODELS_DIR = ROOT_DIR / "models" / "yolo" / "finetuned" / "chess_yolov8s_finetuned"
DATA_DIR = ROOT_DIR / "data" / "recorded_dataset"


class PiecesDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = list(CPIECE2IDX.keys())

    def detect_frames(self, frames):
        piece_detections = []

        for frame in frames:
            piece_dict = self.detect_frame(frame)
            piece_detections.append(piece_dict)

        return piece_detections

    def detect_frame(self, frame):
        # results = self.model.track(frame, persist=False)[0]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            frame_rgb,
            verbose=False,
            imgsz=1024,
            # conf=0.15,
        )[0]
        id_name_dict = results.names

        pieces_dict = {}
        for box in results.boxes:
            # if box.id is None:
            #     continue
            # track_id = int(box.id.tolist()[0])  # Tracker id
            result = box.xyxy.tolist()[0]  # [xmin, ymin, xmax, ymax]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            object_accuracy = box.conf.tolist()[0]

            if not object_cls_name in self.classes:
                continue

            pieces_dict[len(pieces_dict)] = {
                "bbox": result,
                "class_id": object_cls_id,
                "class_name": object_cls_name,
                "accuracy": object_accuracy,
            }

        return pieces_dict

    def draw_bbox(self, frame, piece_dict):
        for idx, piece_data in piece_dict.items():
            # Draw Bounding Boxes
            bbox = piece_data["bbox"]
            x1, y1, x2, y2 = bbox
            class_name = piece_data["class_name"]
            cv2.putText(
                frame,
                f"{class_name}",  # f"Class: {class_name}"
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        return frame


def predict_video(model_path, video_path, output_video_path, input_fps=2, output_fps=2):
    frames = read_video(video_path, fps=input_fps)
    tracker = PiecesDetector(model_path)
    output_frames = []
    for frame in tqdm(frames):
        # Rotate frame 90 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        objects_dict = tracker.detect_frame(frame)
        frame_with_bbox = tracker.draw_bbox(frame, objects_dict)
        output_frames.append(frame_with_bbox)

    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))
    save_video(output_frames, output_video_path, output_fps=output_fps)


if __name__ == "__main__":
    model_path = str(MODELS_DIR / "weights" / "best.pt")
    # model_path = str(MODELS_DIR / "weights" / "last.pt")

    # video_path = str(DATA_DIR / "JSnfTfK__Bc" / "JSnfTfK__Bc_trim.mov")
    # output_video_path = str(DATA_DIR / "JSnfTfK__Bc" / "JSnfTfK__Bc_output.mp4")

    video_path = str(DATA_DIR / "rrPfmSWlAPM" / "rrPfmSWlAPM_trim.mov")
    output_video_path = str(DATA_DIR / "rrPfmSWlAPM" / "rrPfmSWlAPM_output.mp4")

    print("Processing video...")
    print(f"Model path: {model_path}")
    print(f"Video path: {video_path}")
    print(f"Output video path: {output_video_path}")
    predict_video(model_path, video_path, output_video_path)
