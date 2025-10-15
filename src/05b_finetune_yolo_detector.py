import os
import pathlib
import shutil
import torch
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

DATASET_DIR = ROOT_DIR / "data" / "roboflow_chess_dataset"
EPOCHS = 1


def download_dataset():
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("test-wzcdu").project("chess-pieces-detection-camera-pxzh6")
    version = project.version(3)
    dataset = version.download("yolov8")

    print("Original Dataset location: ", dataset.location)
    print("Moving dataset to: ", DATASET_DIR)
    shutil.move(dataset.location, str(DATASET_DIR))


def train_yolo_model():

    # Check if GPU/MPS is available
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load a pre-trained YOLO model (you can choose different sizes: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    model = YOLO("yolov8s.pt")

    # Train the model on the custom dataset
    model.train(
        data=os.path.join(str(DATASET_DIR), "data.yaml"),
        epochs=EPOCHS,
        imgsz=1216,  # Multiple of 32 (1216)
        batch=3,
        device=device,
        project=str(ROOT_DIR / "models" / "yolo" / "finetuned"),
        name="chess_yolov8m_finetuned",
        exist_ok=True,
        # augment=False,
        # plots=False,
        # cache=False,
        # workers=0,
        # val=False,
        degrees=90.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.1,
    )

    # Evaluate the model
    # results = model.val()
    # print(results)


def train_with_cli():
    import subprocess

    cmd = [
        "yolo",
        "train",
        f"data={DATASET_DIR}/data.yaml",
        "model=yolov8s.pt",
        "epochs=1",
        "imgsz=640",
        "batch=1",
        f"project={ROOT_DIR}/models/yolo/finetuned",
        "name=chess_cli_test",
        "exist_ok=True",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    return result.returncode == 0


if __name__ == "__main__":
    # download_dataset()
    train_yolo_model()
