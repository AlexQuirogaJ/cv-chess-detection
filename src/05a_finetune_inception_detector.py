import os
import cv2
import json
import pathlib
import copy
from tqdm import tqdm
from pprint import pprint

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch

import matplotlib.pyplot as plt

from torchsummary import summary
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import seaborn as sns

from model_maps import CPIECE2IDX as CLASS2ID


current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent
# REAL_DATASET_NAME = "real_chess_dataset"
REAL_DATASET_NAME = "recorded_dataset"
REAL_DATASET_DIR = ROOT_DIR / "data" / REAL_DATASET_NAME
TEST_TRAIN_RATIO = 0.2

MLFLOW_EXPERIMENT_NAME = "RecordedChessModel"
MLRUNS_DIR = ROOT_DIR / "data" / "mlruns"

SQUARE_SIZE = 299  # For InceptionV3 compatibility
BOARD_PADDING = 0  # Percentage of padding around the square

BATCH_SIZE = 128
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3


def fen2arr(fen):
    # Only use the piece placement part
    fen_piece_placement = fen.split(" ")[
        0
    ]  # rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2
    board_arr = np.empty((8, 8), dtype=object)
    rows = fen_piece_placement.split("/")
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                for _ in range(int(char)):
                    board_arr[i, col] = "0"  # empty square
                    col += 1
            else:
                board_arr[i, col] = char
                col += 1
    return board_arr


class ChessSquaredDataset(Dataset):
    def __init__(self, dataset_dir, class2id, video_ids=None, augment_scaling: int = 4):
        self.dataset_dir = dataset_dir
        self.class2id = class2id
        self.augment_scaling = augment_scaling

        # Load all video IDs
        if not video_ids:
            video_ids = []
            for video_dir in dataset_dir.glob("*"):
                if not video_dir.is_dir() or video_dir.name.startswith("."):
                    continue
                video_ids.append(video_dir.name)
        self.video_ids = video_ids

        # Load board image paths and FEN labels
        board_img_paths = []
        fen_and_move_labels = []
        for video_id in self.video_ids:
            imgs_dir = dataset_dir / video_id / "squared"
            fen_path = dataset_dir / video_id / "fen.csv"

            for img_file in imgs_dir.glob("*.jpg"):
                img_name = img_file.stem
                move_number = int(
                    img_name.split("_")[-1]
                )  # Extract move number from filename
                if self.augment_scaling > 1:
                    for _ in range(self.augment_scaling):
                        board_img_paths.append(str(img_file))
                        fen_and_move_labels.append((fen_path, move_number))
                else:
                    board_img_paths.append(str(img_file))
                    fen_and_move_labels.append((fen_path, move_number))

        self.samples = []

        for img_path, (fen_path, move_number) in zip(
            board_img_paths, fen_and_move_labels
        ):
            # 64 squares per image
            for row in range(8):
                for col in range(8):
                    self.samples.append((img_path, fen_path, move_number, row, col))

        # self.transform = transforms.ToTensor() # Convert images to tensors (0-255 -> 0-1)
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(5),
                transforms.RandomResizedCrop(SQUARE_SIZE, scale=(0.8, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
        )

        # self.transforms = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.RandomAffine(
        #     degrees=5,
        #     translate=(0.1, 0.1),
        #     scale=(0.8, 1),
        #     shear=1
        # ),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fen_path, move_number, row, col = self.samples[idx]

        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Crop the board padding
        h, w = img.shape[:2]
        start_h = int(h * BOARD_PADDING // 100)
        start_w = int(w * BOARD_PADDING // 100)
        img = img[start_h : h - start_h, start_w : w - start_w]

        # Crop the specific square
        img = cv2.resize(img, (8 * SQUARE_SIZE, 8 * SQUARE_SIZE))
        square_img = img[
            row * SQUARE_SIZE : (row + 1) * SQUARE_SIZE,
            col * SQUARE_SIZE : (col + 1) * SQUARE_SIZE,
        ]

        # Load the label
        with open(fen_path, "r") as f:
            df_fen = pd.read_csv(f)

        board_fen = df_fen.loc[df_fen["index"] == move_number, "fen"].iloc[0]
        square_label = fen2arr(board_fen)[row, col]  # move_number is 1-indexed

        # Move the channel dimension to the front and apply transforms
        square_img_tensor = self.transforms(square_img)  # Convert to tensor (3, 60, 60)

        # Convert label to class ID
        square_label = self.class2id.get(square_label, -1)  # Use -1 for unknown labels

        return square_img_tensor, square_label


class InceptionV3Modified(nn.Module):

    def __init__(self, num_classes=13):
        super(InceptionV3Modified, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        num_ftrs = self.inception.fc.in_features

        # Freeze model up to Mixed_7c
        for name, parameter in self.inception.named_parameters():
            parameter.requires_grad = False
            if "Mized_7c" in name:
                break

        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(CLASS2ID)),
        )

    def forward(self, x):
        out = self.inception(x)
        if isinstance(out, tuple):
            out = out[0]  # Use main output, ignore aux
        return out


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()  # Set model to training mode

    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Log progress every 10 batches
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            mlflow.log_metric("train_loss", loss, step=current)

    avg_train_loss = train_loss / len(dataloader)
    train_accuracy = correct / size

    return avg_train_loss, train_accuracy


def validation_loop(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_val_loss = val_loss / num_batches
    val_accuracy = correct / size

    print(f"Val Error: Accuracy: {avg_val_loss:.4f}, Avg loss: {avg_val_loss:.4f} \n")

    return avg_val_loss, val_accuracy


def train_model(
    model, train_dataloader, val_dataloader, device, num_epochs=10, learning_rate=1e-3
):
    loss_fn = (
        nn.CrossEntropyLoss()
    )  # Applies softmax so no need to do add softmax to the model output
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # Log hyperparameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)

    # Best model accuracy
    best_val_accuracy = 0.0

    # Create an input example for signature inference
    input_example = np.random.randn(1, 3, SQUARE_SIZE, SQUARE_SIZE).astype(np.float32)

    # for epoch in tqdm(range(num_epochs), desc="Training"):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1:>4d}/{num_epochs:>4d}\n-------------------------------")

        avg_train_loss, avg_train_accuracy = train_loop(
            train_dataloader, model, loss_fn, optimizer, device
        )
        avg_val_loss, avg_val_accuracy = validation_loop(
            val_dataloader, model, loss_fn, device
        )

        print(f"Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        print(
            f"Avg Train Accuracy: {avg_train_accuracy:.4f}, Avg Val Accuracy: {avg_val_accuracy:.4f}\n"
        )

        # Log metrics for each epoch
        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("train_accuracy", avg_train_accuracy, step=epoch)
        mlflow.log_metric("val_accuracy", avg_val_accuracy, step=epoch)

        # Save best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            model_cpu = copy.deepcopy(model).to(
                "cpu"
            )  # Move model to CPU before logging
            mlflow.pytorch.log_model(
                model_cpu, name="best_model", input_example=input_example
            )
            print(
                f"Best model updated at epoch {epoch+1} with val accuracy {best_val_accuracy:.4f}"
            )

    # Log the trained model
    model_cpu = copy.deepcopy(model).to("cpu")  # Move model to CPU before logging
    mlflow.pytorch.log_model(model_cpu, name="last_model", input_example=input_example)

    print("Done!")


def test_model(model, test_dataloader, device, class_names):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print(classification_report(all_labels, all_preds, target_names=class_names))
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


if __name__ == "__main__":

    # Define complete dataset
    complete_dataset = ChessSquaredDataset(
        dataset_dir=REAL_DATASET_DIR, class2id=CLASS2ID, augment_scaling=4
    )

    # Split dataset into training and testing
    train_size = int(TEST_TRAIN_RATIO * len(complete_dataset))
    test_size = len(complete_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    test_dataset, train_dataset = random_split(
        complete_dataset, [train_size, test_size], generator=generator
    )

    print(f"Number of samples: {len(complete_dataset)}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    train_imgs_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"Batch image shape: {train_imgs_batch.shape}")
    print(f"Batch labels shape: {len(train_labels_batch)}")

    print(train_labels_batch)

    def finetune_inceptionv3():
        # Create the model
        model = InceptionV3Modified(num_classes=len(CLASS2ID))
        summary(
            model, input_size=(3, SQUARE_SIZE, SQUARE_SIZE)
        )  # (Channels, Height, Width)

        # Training
        print("Starting training...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        # Set save directory
        print("Mlruns dir: ", MLRUNS_DIR)
        mlflow.set_tracking_uri("file://" + str(MLRUNS_DIR))

        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Training and logging
        with mlflow.start_run():

            # Create and prepare model
            model = model.to(device)

            # Train the model
            train_model(
                model,
                train_dataloader,
                test_dataloader,
                device=device,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
            )

    def test_inceptionv3():
        # Select device for inference
        print("Starting Prediction...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        # Load the model from mlflow
        RUN_ID = "2e0c9086863844c29195fe15c81879d2"
        mlflow.set_tracking_uri("file://" + str(MLRUNS_DIR))
        model = mlflow.pytorch.load_model("runs:/" + RUN_ID + "/best_model")
        # model = mlflow.pytorch.load_model("runs:/" + RUN_ID + "/last_model")

        # Evaluate on the test set
        test_model(model, test_dataloader, device, class_names=list(CLASS2ID.keys()))

    # finetune_inceptionv3()
    test_inceptionv3()
