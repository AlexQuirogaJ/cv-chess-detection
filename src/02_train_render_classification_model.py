import os
import cv2
import numpy as np
import json
import mlflow
import torch
import torch.nn as nn
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import seaborn as sns

import mlflow.pytorch
import mlflow

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pprint import pprint


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = pathlib.Path(BASE_DIR).parent

DATASET_NAME = "render_chess_dataset"
DATASET_DIR = ROOT_DIR / "data" / DATASET_NAME

MLFLOW_EXPERIMENT_NAME = "RenderChessModel"
MLRUNS_DIR = ROOT_DIR / "data" / "mlruns"

# Trained model run id
RENDER_MODEL_RUN_ID = "0c2d1755485045b28e5048e624b5d6d4"


class Chess2dRenderedDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, class2id):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir

        self.board_img_paths = [
            os.path.join(imgs_dir, img) for img in sorted(os.listdir(imgs_dir))
        ]
        self.labels_paths = [
            os.path.join(labels_dir, label) for label in sorted(os.listdir(labels_dir))
        ]
        self.samples = []

        for img_path, label_path in zip(self.board_img_paths, self.labels_paths):
            # 64 squares per image
            for row in range(8):
                for col in range(8):
                    self.samples.append((img_path, label_path, row, col))

        self.transform = (
            transforms.ToTensor()
        )  # Convert images to tensors (0-255 -> 0-1)

        self.class2id = class2id  # Dictionary mapping class names to IDs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, row, col = self.samples[idx]

        # Constants
        SQUARE_SIZE = 60
        BOARD_PADDING = 0  # Percentage of padding around the square

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
        with open(label_path, "r") as f:
            label_arr = np.loadtxt(f, delimiter=",", dtype=str)

        square_label = label_arr[row, col]

        # Move the channel dimension to the front
        # square_img = np.transpose(square_img, (1, 2, 0)) # Dont need (transoform already does this)

        square_img_tensor = self.transform(square_img)  # Convert to tensor (3, 60, 60)

        # Convert label to class ID
        square_label = self.class2id.get(square_label, -1)  # Use -1 for unknown labels

        return square_img_tensor, square_label


class SmallAlexNet(nn.Module):
    def __init__(self, num_classes=13, input_size=(3, 60, 60)):
        super(SmallAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=1, padding=2),  # (54, 54, 96)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (26, 26, 96)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # (26, 26, 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (12, 12, 256)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # (12, 12, 384)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # (12, 12, 384)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # (12, 12, 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (5, 5, 256) = 6400 features
        )

        with torch.no_grad():
            # Calculate the output size after the feature extractor
            sample_input = torch.zeros(1, *input_size)
            output_features_shape = self.features(sample_input).shape[1:]

        output_features_size = np.prod(output_features_shape)
        print("Output features size:", output_features_shape)
        print("Output features size:", output_features_size)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(output_features_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


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

    avg_train_loss = train_loss / len(dataloader)
    train_accuracy = correct / size

    return avg_train_loss, train_accuracy


def test_loop(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # Get logits (no probabilities)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_test_loss = test_loss / num_batches
    test_accuracy = correct / size

    print(
        f"Test Error: Accuracy: {avg_test_loss:.4f}, Avg loss: {avg_test_loss:.4f} \n"
    )

    return avg_test_loss, test_accuracy


def train_model(
    model, train_dataloader, test_dataloader, device, num_epochs=10, learning_rate=1e-3
):
    loss_fn = (
        nn.CrossEntropyLoss()
    )  # Applies softmax so no need to do add softmax to the model output
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Log hyperparameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)

    # Best model accuracy
    best_test_accuracy = 0.0

    # for epoch in tqdm(range(num_epochs), desc="Training"):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1:>4d}/{num_epochs:>4d}\n-------------------------------")

        avg_train_loss, avg_train_accuracy = train_loop(
            train_dataloader, model, loss_fn, optimizer, device
        )
        avg_test_loss, avg_test_accuracy = test_loop(
            test_dataloader, model, loss_fn, device
        )

        print(
            f"Avg Train Loss: {avg_train_loss:.4f}, Avg Test Loss: {avg_test_loss:.4f}"
        )
        print(
            f"Avg Train Accuracy: {avg_train_accuracy:.4f}, Avg Test Accuracy: {avg_test_accuracy:.4f}\n"
        )

        # Log metrics for each epoch
        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("avg_test_loss", avg_test_loss, step=epoch)
        mlflow.log_metric("train_accuracy", avg_train_accuracy, step=epoch)
        mlflow.log_metric("test_accuracy", avg_test_accuracy, step=epoch)

        # Save best model
        if avg_test_accuracy > best_test_accuracy:
            best_test_accuracy = avg_test_accuracy
            mlflow.pytorch.log_model(model, "best_model")
            print(
                f"Best model updated at epoch {epoch+1} with test accuracy {best_test_accuracy:.4f}"
            )

    # Log the trained model
    mlflow.pytorch.log_model(model, "last_model")

    print("Done!")


if __name__ == "__main__":

    # Dataset dir and load labels mapping
    labels_mapping_path = DATASET_DIR / "labels_mapping.json"

    with open(labels_mapping_path, "r") as f:
        labels_map = json.load(f)

    # Train, test and validation dataset instantiation
    train_imgs_dir = DATASET_DIR / "train" / "images"
    train_labels_dir = DATASET_DIR / "train" / "labels"
    train_dataset = Chess2dRenderedDataset(
        imgs_dir=train_imgs_dir,
        labels_dir=train_labels_dir,
        class2id=labels_map["class2id"],
    )

    test_imgs_dir = DATASET_DIR / "test" / "images"
    test_labels_dir = DATASET_DIR / "test" / "labels"
    test_dataset = Chess2dRenderedDataset(
        imgs_dir=test_imgs_dir,
        labels_dir=test_labels_dir,
        class2id=labels_map["class2id"],
    )

    validation_imgs_dir = DATASET_DIR / "validation" / "images"
    validation_labels_dir = DATASET_DIR / "validation" / "labels"
    validation_dataset = Chess2dRenderedDataset(
        imgs_dir=validation_imgs_dir,
        labels_dir=validation_labels_dir,
        class2id=labels_map["class2id"],
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Number of validation samples: {len(validation_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=True)

    print(f"Number of training batches: {len(train_dataloader)}")
    train_imgs_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"Batch image shape: {train_imgs_batch.shape}")
    print(f"Batch labels shape: {len(train_labels_batch)}")
    # print(train_labels_batch)

    # Set save directory
    print("Mlruns dir: ", MLRUNS_DIR)
    mlflow.set_tracking_uri("file://" + str(MLRUNS_DIR))

    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def train():

        # Create model
        print("Creating model...")
        model = SmallAlexNet(num_classes=13, input_size=(3, 60, 60))
        summary(model)

        # Test a forward pass
        print("Testing a forward pass...")
        X, y = next(iter(train_dataloader))
        print(f"Input batch shape: {X.shape} {type(X)}")
        print(f"Labels batch shape: {len(y)} {type(y[0])}")

        # print(y[0:3])
        y_pred = model(X)
        print(f"Predicted batch shape: {y_pred.shape}")
        print(y_pred[0])

        # Training
        print("Starting training...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

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
                num_epochs=5,
                learning_rate=1e-3,
            )

    def test():

        # Load the best model
        run_id = RENDER_MODEL_RUN_ID
        model = mlflow.pytorch.load_model("runs:/" + run_id + "/best_model")

        all_preds = []
        all_labels = []

        print("Starting testing...")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        for batch, (X_test, y_test) in enumerate(test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            y_pred = y_pred.argmax(dim=1)

            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

        # Compute accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )

        cm = confusion_matrix(all_labels, all_preds)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        print("Classification report: ")
        print(classification_report(all_labels, all_preds))

        plt.figure(figsize=(10, 7))

        id2label = labels_map["id2class"]
        id2label = {int(k): v for k, v in id2label.items()}  # Convert keys to int
        class_names = [id2label[i] for i in sorted(id2label.keys())]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # train()
    test()
