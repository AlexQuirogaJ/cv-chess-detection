import os
import json
import pathlib
import shutil
import subprocess
import concurrent.futures
from pprint import pprint
from copy import deepcopy

import concurrent

import torch
import numpy as np
import chess
import matplotlib.pyplot as plt
import cv2
import mlflow
import chess.svg
import pandas as pd
import cairosvg
from tqdm import tqdm
from loguru import logger
from pytubefix import YouTube

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

RENDER_DATASET_NAME = "render_chess_dataset"
RENDER_DATASET_DIR = ROOT_DIR / "data" / RENDER_DATASET_NAME

MLRUNS_DIR = ROOT_DIR / "data" / "mlruns"
# RENDER_MODEL_RUN_ID = "a8ccc1d4403b4bdebc379b2926cda62b"
RENDER_MODEL_RUN_ID = "0c2d1755485045b28e5048e624b5d6d4"

DOWNLOADED_VIDEOS_DIR = ROOT_DIR / "data" / "videos"
REAL_DATASET_NAME = "real_chess_dataset"
REAL_DATASET_DIR = ROOT_DIR / "data" / REAL_DATASET_NAME

from video_settings import GENERAL_DATASET_VIDEOS_SETTINGS as DATASET_VIDEOS_SETTINGS

# from video_settings import FIDE_DATASET_VIDEOS_SETTINGS as DATASET_VIDEOS_SETTINGS


def pull_video(video_id, save_dir):
    out_fn = f"{save_dir}/{video_id}.mp4"
    if os.path.exists(out_fn):
        return

    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    print(f"Downloading video: {yt.title}")

    # Try 1080p mp4 video only stream or 720p if not available
    streams = yt.streams.filter(
        progressive=False, only_video=True, res="1080p", mime_type="video/mp4"
    )
    if not streams:
        streams = yt.streams.filter(
            progressive=False, only_video=True, res="720p", mime_type="video/mp4"
        )

    try:
        stream = streams[0]
    except IndexError:
        raise ValueError("No stream for 1080p or 720p found.")
    # stream = yt.streams.get_highest_resolution() # Progressive stream has low resolution

    stream.download()
    fn = stream.default_filename
    shutil.move(fn, out_fn)


def load_id2label_map(dataset_dir):
    labels_mapping_path = dataset_dir / "labels_mapping.json"

    with open(labels_mapping_path, "r") as f:
        labels_map = json.load(f)

    id2label = labels_map["id2class"]

    # Convert keys to int
    id2label = {int(k): v for k, v in id2label.items()}

    return id2label


class GTChessboardPredictor:
    def __init__(
        self,
        id2label_map,
        square_size=60,
        mlruns_dir="../data/mlruns",
        mlrun_uri=None,
        run_id=None,
        initial_state=None,
        from_beginning=True,
        best_model=True,
    ):
        # mlruns_dir (str, optional): The location to store the MLflow runs.
        #     Defaults to "../data/mlruns".
        # mlrun_uri (str, optional): The URI for the MLflow run.
        #     Defaults to None.
        # run_id (str, optional): The ID of the MLflow run.
        #     Defaults to None.

        self.mlruns_dir = mlruns_dir
        self.mlrun_uri = mlrun_uri
        self.run_id = run_id
        self.id2label_map = id2label_map  # Map that relates class indices to labels
        self.square_size = square_size
        self.initial_state = initial_state
        self.from_beginning = from_beginning
        self.best_model = best_model

        if self.initial_state:
            self.from_beginning = False

        # # Store all possible prediction for each square
        # self.predictions = [[[] for _ in range(8)] for _ in range(8)]

        # Chess board
        self.board = chess.Board()
        self.black_pov = False
        self.prev_board = None
        self.board_arr = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            ["0", "0", "0", "0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0", "0", "0", "0"],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]
        self.n_moves = 0

        # Set mlruns dir
        mlflow.set_tracking_uri("file://" + self.mlruns_dir)

        # Load model
        if self.mlrun_uri:
            mlflow.set_tracking_uri(self.mlrun_uri)
        if self.run_id:
            if self.best_model:
                self.model = mlflow.pytorch.load_model(
                    "runs:/" + self.run_id + "/best_model"
                )
            else:
                self.model = mlflow.pytorch.load_model(
                    "runs:/" + self.run_id + "/last_model"
                )

    def get_board_loss(self, pred_probs, legal_board_arr):
        # Compute the loss for the predicted board arrangement
        loss = 0.0
        for row in range(8):
            for col in range(8):
                piece = legal_board_arr[row, col]
                prob = pred_probs[row][col].get(
                    piece, 0
                )  # Get probability for the piece
                loss -= np.log(prob + 1e-8)  # Add log probability to the loss
        return loss

    def get_valid_move_and_board(self, board, pred_probs, pred_board_arr):

        # Get current board array
        current_fen = board.fen()
        current_board_arr = self.fen2arr(current_fen)

        # Check if no moves
        if np.array_equal(pred_board_arr, current_board_arr):
            loss = self.get_board_loss(pred_probs, current_board_arr)
            return None, current_board_arr, loss

        # Get score for no move
        no_move_loss = self.get_board_loss(pred_probs, current_board_arr)
        legal_move = None
        legal_board_arr = current_board_arr
        min_loss = no_move_loss

        # Find minimum loss for each legal move
        board_copy = board.copy()
        for move in board.legal_moves:
            board_copy.push(move)
            move_board_arr = self.fen2arr(board_copy.fen())
            loss = self.get_board_loss(pred_probs, move_board_arr)
            if loss < min_loss:
                min_loss = loss
                legal_move = move
                legal_board_arr = move_board_arr
            board_copy.pop()

        return legal_move, legal_board_arr, float(min_loss)

    def flip_board_arr(self, board_arr):
        # inverted_board = [row[::-1] for row in board_arr[::-1]]
        return np.rot90(board_arr, 2)

    def flip_prob_preds(self, prob_preds):
        inverted_prob_preds = [row[::-1] for row in prob_preds[::-1]]
        return inverted_prob_preds

    def predict(
        self,
        gt_board_image: np.ndarray,
        device: torch.device,
        logger: logger,
        use_previous_moves: bool = False,
        verbose: bool = False,
    ):

        # print("Predicting...")
        # Predict without previous moves
        if not use_previous_moves:
            predicted_board_arr, _ = self.predict_gt_board(gt_board_image, device)
            fen = self.arr2fen(predicted_board_arr)
            self.board = chess.Board(fen)
            self.prev_board = None
            return fen, True

        # Predict with previous moves
        predicted_board_arr, pred_probs = self.predict_gt_board(gt_board_image, device)

        # Check if the predicted board is in black perspective
        if self.from_beginning and self.n_moves == 0:
            if predicted_board_arr[0][0] == "R":
                # Black pov
                self.black_pov = True
                logger.info("Predicted board is in black perspective.")

        if self.black_pov:
            predicted_board_arr = self.flip_board_arr(predicted_board_arr)
            pred_probs = self.flip_prob_preds(pred_probs)

        if not self.from_beginning and self.n_moves == 0:

            if self.initial_state:
                fen = self.arr2fen(self.initial_state)
                fen += " w KQkq - 0 1"  # Add castling rights
            else:
                fen = self.arr2fen(predicted_board_arr)
                fen += " w KQkq - 0 1"  # Add castling rights

            self.board = chess.Board(fen)
            self.prev_board = None
            self.n_moves += 1

            final_valid_move = None
            changed = False
            final_min_loss = None
            reject_prev_pred = False
            return fen, final_valid_move, changed, final_min_loss, reject_prev_pred

        current_board_move, current_legal_board_arr, current_min_loss = (
            self.get_valid_move_and_board(self.board, pred_probs, predicted_board_arr)
        )
        final_valid_move = current_board_move
        final_legal_board_arr = current_legal_board_arr
        final_min_loss = current_min_loss

        reject_prev_pred = False
        if self.prev_board and current_min_loss > 10.0:
            # print("Calculate prev board")

            prev_valid_move, prev_legal_board_arr, prev_min_loss = (
                self.get_valid_move_and_board(
                    self.prev_board, pred_probs, predicted_board_arr
                )
            )

            logger.info(
                f"Move: {current_board_move}, Prev Move: {prev_valid_move}, current loss: {current_min_loss}, prev loss {prev_min_loss}"
            )
            logger.info(f"Current board: {self.board.fen()}")
            logger.info(f"Previous board: {self.prev_board.fen()}")

            if prev_min_loss < current_min_loss:
                reject_prev_pred = True
                final_valid_move = prev_valid_move
                final_legal_board_arr = prev_legal_board_arr
                final_min_loss = prev_min_loss

                logger.warning("Previous board had a better move.")

        if verbose:
            logger.info("Valid move: ", final_valid_move)
            logger.info("Legal board: ", final_legal_board_arr)

        changed = False
        if final_valid_move:
            changed = True
            if reject_prev_pred:
                self.board = self.prev_board.copy()
                self.board.push(final_valid_move)
                logger.warning("Rolled back to previous state and applied new move.")
            else:
                self.prev_board = self.board.copy()
                self.board.push(final_valid_move)
                # print("Applied move to current state.")

        else:
            if verbose:
                print("No move, the board is unchanged.")

        fen = self.arr2fen(final_legal_board_arr)
        return fen, final_valid_move, changed, final_min_loss, reject_prev_pred

    def arr2fen(self, pred_board_arr):
        fen = ""
        for row in pred_board_arr:
            empty_count = 0
            for square in row:
                if square == "0":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen += str(empty_count)  # Add empty square count
                        empty_count = 0
                    fen += square if len(square) == 1 else "?"  # fallback for unknown
            if empty_count > 0:
                fen += str(empty_count)
            fen += "/"
        return fen[:-1]  # Remove trailing slash

    def fen2arr(self, fen):
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

    def predict_gt_board(self, gt_board_img: np.ndarray, device: torch.device):

        square_imgs = []
        resized_gt_board_img = cv2.resize(
            gt_board_img, (8 * self.square_size, 8 * self.square_size)
        )

        for row in range(8):
            for col in range(8):
                # Crop the specific square
                square_img = resized_gt_board_img[
                    row * self.square_size : (row + 1) * self.square_size,
                    col * self.square_size : (col + 1) * self.square_size,
                ]

                # Preprocess the square
                square_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
                square_img = square_img / 255.0
                square_img = square_img.transpose((2, 0, 1))  # (3, 60, 60)
                square_imgs.append(square_img)

        gt_board_squares_t = torch.tensor(
            np.array(square_imgs), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            batch_preds_logits_t = self.model(
                gt_board_squares_t
            )  # (batch size, num_classes) = (64, 13)
            batch_preds_probs_n = batch_preds_logits_t.softmax(dim=1).cpu().numpy()
            preds_n = batch_preds_probs_n.argmax(axis=1)

        # Fill in the probabilities for each square
        pred_board_arr: np.ndarray = np.empty((8, 8), dtype=object)
        pred_probs: list[list[dict]] = [[{} for _ in range(8)] for _ in range(8)]

        for i in range(64):
            row, col = i // 8, i % 8
            # Map the predicted class to the corresponding label
            chosen_square_pred_label = self.id2label_map.get(preds_n[i], "Unknown")
            # Add to predictions array
            pred_board_arr[row, col] = chosen_square_pred_label

            square_pred_probs_dict = {
                self.id2label_map.get(i, "Unknown"): prob
                for i, prob in enumerate(batch_preds_probs_n[i])
            }

            # Save all predictions probabilities for each square
            pred_probs[row][col] = square_pred_probs_dict

        return pred_board_arr, pred_probs


def line_flattener(line):
    # Replace consecutive '1's with their count in each line of the FEN string
    linenew = line
    for i in range(9, 1, -1):
        linenew = linenew.replace("1" * i, str(i))
    return linenew


def save_video(output_video_frames, output_video_path, fps=2):
    output_dir = output_video_path.rsplit("/", 1)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )  # Sets the video codec using OpenCV's VideoWriter_fourcc
    # MJPG is a codec (Motion JPEG) and * unpacks the string so it's passed as separate characters to the function
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,  #
        fps,  # 24 FPS
        (
            output_video_frames[0].shape[1],
            output_video_frames[0].shape[0],
        ),  # (width, height)
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()


class VideoBoardExtractor:
    def __init__(
        self,
        video_path: str,
        gt_board_loc=[20, 415, 440, 835],
        irl_board_loc=[420, 720, 350, 1280],
        time_range=None,
        step_sec=1.0,
        store_masks=False,
        store_gt_boards=False,
        store_gt_video=False,
        store_irl_boards=False,
        store_irl_video=False,
        store_pred_boards=False,
        store_pred_video=False,
        save_img_freq=False,
        at_first_gt_frame=True,
        purge_output_dir=True,
        store_dir="../data/processed",
        gt_board_predictor=None,
        save_per_frame=False,
        continues_saving=False,
    ):
        """Object for extracting features from a chess video mp4 file.
        The video is expected to have a ground truth board placed at
        the top middle of the screen and the live in real life board
        at the bottom middle of the screen. Video should be in mp4 format.

        Args:
            video_fn (str): The video filename .mp4 format
            gt_board_loc (list, optional): The location of the ground
                truth board. Defaults to [20, 415, 440, 835].
            irl_board_loc (list, optional): The location of the real
                life board. Defaults to [420, -1, 350, -350].
            store_masks (bool, optional): Option to store a masked image
                of the video for each frame. Defaults to False.
            store_gt_boards (bool, optional): Option to store the ground truth
                boards image from the video. Defaults to False.
            store_irl_video (bool, optional): Option to store the irl board as
                a video file.
            store_irl_boards (bool, optional): Option to store the real life
                board imags. Defaults to False.
            save_img_freq (int or bool):
                Will save the image of the video frame every `save_img_freq` to
                the `img` data directory
            at_first_gt_frame (bool, optional): If False will try to automatically
                detect the first frame where there is a ground truth board.
            purge_output_dir (bool, optional): If True will delete everything in the
                output directory for the video.
            store_dir (str, optional): The location to store the outputs of
                the processing. Will be placed in a subfolder with the video_id.
                Defaults to "../data/processed".
            gt_board_predictor (GTChessboardPredictor, optional): The ground truth board predictor.
        """
        # Logging
        self.logger = logger

        # Video extraction
        self.time_range = time_range
        self.step_sec = step_sec
        self.video_path = video_path
        self.vidcap = None
        self.first_frame_img = None
        self.avg_frame_color = {}
        self.last_frame_color = None

        # FEN prediction utilities
        self.frame_change = None
        self.at_first_gt_frame = at_first_gt_frame
        self.gt_board_loc = gt_board_loc
        self.last_fen = None
        self.is_first_frame = True

        # FEN prediction model
        self.move_number = 0
        self.fens = []
        self.fen_predictor = gt_board_predictor

        self.min_losses = []

        # Result storage
        self.store_dir = store_dir
        self.video_name = video_path.split("/")[-1].strip(".mp4")
        self.full_store_dir = f"{self.store_dir}/{self.video_name}"
        self.save_per_frame = save_per_frame
        self.store_masks = store_masks
        self.store_gt_boards = store_gt_boards
        self.store_gt_video = store_gt_video
        self.store_pred_boards = store_pred_boards
        self.store_pred_video = store_pred_video
        self.irl_board_loc = irl_board_loc
        self.store_irl_boards = store_irl_boards
        self.store_irl_video = store_irl_video
        self.save_img_freq = save_img_freq
        self.continues_saving = continues_saving

        log_file_path = f"{self.full_store_dir}/process.log"
        self.logger.add(log_file_path)
        # Empty file if it exists
        if os.path.exists(log_file_path):
            open(log_file_path, "w").close()

        if store_irl_boards or store_gt_boards or store_masks or store_irl_video:
            self.make_dirs(purge_output_dir)

        self.logger.info(f"Created video extractor for {self.video_path}")
        self.logger.info(f"IRL Board Location: {irl_board_loc}")
        self.logger.info(f"GT Board Location: {gt_board_loc}")

    def load_videocap(self):
        self.logger.info("Loading video capture")
        self.vidcap = cv2.VideoCapture(self.video_path)
        self.frame_count = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.vidcap.get(cv2.CAP_PROP_FPS)

    def save_video_img(self, frame, image):
        if frame % self.save_img_freq == 0:
            img_fn = f"{self.full_store_dir}/imgs/{frame}.png"
            cv2.imwrite(img_fn, image)

    def process_video(self):
        self.logger.info("Proccessing video capture")

        # Load video
        self.load_videocap()

        # Time range
        if self.time_range:
            # Get the start and end times in seconds (format 02:21)
            # start_time = sum(x * int(t) for x, t in zip([60, 1], self.time_range[0].split(":")))
            # end_time = sum(x * int(t) for x, t in zip([60, 1], self.time_range[1].split(":")))
            # print(f"Extracting frames from {self.time_range[0]} to {self.time_range[1]} ....")

            # Get the start and end times in seconds (format 02:21:00)
            start_time = sum(
                x * int(t) for x, t in zip([60, 1, 0.01], self.time_range[0].split(":"))
            )
            end_time = sum(
                x * int(t) for x, t in zip([60, 1, 0.01], self.time_range[1].split(":"))
            )
            print(
                f"Extracting frames from {self.time_range[0]} to {self.time_range[1]} ...."
            )
        else:
            start_time = 0
            end_time = self.frame_count / self.fps
            print(f"Extracting frames from start to end of video: {self.video_name}")

        # Select device for prediction model
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        self.logger.info(f"Using device: {device}")

        # Seek to start time
        self.vidcap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # Get the frames for the specified range
        total_seconds = end_time - start_time
        max_frame = int(total_seconds / self.step_sec)
        pbar = tqdm(total=max_frame)
        current_time = start_time
        frame = 0
        while current_time <= end_time:
            success, image = self.vidcap.read()
            if not success:
                pbar.close()
                break

            _ = self.process_frame(image, frame, device)

            # minute = int(current_time // 60)
            # seconds = int(current_time % 60)
            # milliseconds = int((current_time * 1000) % 1000)
            # cv2.imwrite(str(frames_dir / f"frame_{minute:03d}_{seconds:02d}_{milliseconds:03d}.jpg"), frame)

            # Skip to the next frame
            current_time += self.step_sec
            pbar.update(1)
            frame += 1
            self.vidcap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)

        self.vidcap.release()
        pbar.close()

        if self.store_gt_video and self.store_gt_boards:
            self.video_from_folder_images(f"{self.full_store_dir}/gt/", fps=1)

        if self.store_pred_video and self.store_pred_boards:
            self.video_from_folder_images(f"{self.full_store_dir}/pred/", fps=1)

        if self.store_irl_video and self.store_irl_boards:
            self.video_from_folder_images(f"{self.full_store_dir}/irl/", fps=1)

    def video_from_folder_images(self, folder_path, fps=2):

        snapshots = []
        for img_path in sorted(pathlib.Path(folder_path).iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            iter_number = int(img_path.stem.split("_")[-1])
            img = cv2.imread(str(img_path))

            # Add frame number as text
            show_text = f"Move: {iter_number}"
            if self.save_per_frame:
                show_text = f"Frame: {iter_number}"
            cv2.putText(
                img,
                show_text,
                (20, 40),  # Position of the text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            snapshots.append(img)

        if not snapshots:
            self.logger.warning("No frames found in folder")
            return

        output_video_path = f"{folder_path}/{self.video_name}.mp4"
        save_video(snapshots, output_video_path, fps)

    def get_average_color(self, img, frame):
        self.logger.info("Getting the average color of frame")

        average = img.mean(axis=0).mean(axis=0)

        self.avg_frame_color[frame] = average

        if self.last_frame_color is None:
            self.last_frame_color = np.mean(average)
            return

        self.frame_change = np.mean(average) - self.last_frame_color
        if self.frame_change > 50:
            self.at_first_gt_frame = True
            # print(f'AGHHHH - FRAME {frame}')
        self.last_frame_color = np.mean(average)

    def save_fen_csv(self):
        pd.DataFrame(self.fens).reset_index().to_csv(f"{self.full_store_dir}/fen.csv")

    def save_min_losses_csv(self):
        pd.DataFrame(self.min_losses).reset_index().to_csv(
            f"{self.full_store_dir}/min_losses.csv"
        )

    @logger.catch
    def extract_fen(self, gt_board, frame, move_number, device):
        try:
            fen, valid_move, changed, min_loss, reject_prev_pred = (
                self.fen_predictor.predict(
                    gt_board, device, self.logger, use_previous_moves=True
                )
            )

            self.logger.info(
                f"Nº {move_number}, Frame {frame}, Move: {valid_move}, Changed: {changed}, Min Loss: {min_loss}"
            )

            if reject_prev_pred:
                self.logger.warning(f"Rejecting previous prediction at frame {frame}")
                if self.fens and self.min_losses:
                    del self.fens[-1]
                    del self.min_losses[-1]
                    self.move_number -= 1
                else:
                    self.logger.warning(
                        f"Cannot reject previous prediction at frame {frame}"
                    )

            if changed or frame == 0:
                flat_fen = "/".join([line_flattener(line) for line in fen.split("/")])
                self.fens.append({"frame": frame, "fen": flat_fen})
                self.min_losses.append({"frame": frame, "min_loss": min_loss})

                # Check no diverge in loss
                if min_loss > 120:
                    self.logger.warning(f"Loss diverged at frame {frame}")

            if changed or frame == 0 or reject_prev_pred:
                self.save_fen_csv()
                self.save_min_losses_csv()

        except Exception as e:
            logger.warning("Could not detect the FEN at this frame")
            logger.warning(f"Exception thrown {e}")
            return False, True
        return True, changed

    def make_dirs(self, purge_output_dir):
        self.logger.info("Making storage directories")
        dirs_to_make = []
        dirs_to_make.append(f"{self.full_store_dir}/gt/")
        dirs_to_make.append(f"{self.full_store_dir}/irl/")
        dirs_to_make.append(f"{self.full_store_dir}/mask/")
        dirs_to_make.append(f"{self.full_store_dir}/imgs/")
        dirs_to_make.append(f"{self.full_store_dir}/pred/")

        for d in dirs_to_make:
            if os.path.exists(d) and purge_output_dir:
                shutil.rmtree(d)
            if not os.path.exists(d):
                os.makedirs(d)

    def process_frame(self, image, frame, device):
        if frame == 1 and not self.at_first_gt_frame:
            self.first_frame_img = image.copy()
        if self.at_first_gt_frame is False:
            self.get_average_color(image, frame)
        self.this_frame_img = image.copy()
        self.masked_img, self.gt_board, self.irl_board = self.extract_gt_board(
            image, self.gt_board_loc, self.irl_board_loc
        )

        succeed, fen_changed = self.extract_fen(
            self.gt_board, frame, self.move_number, device
        )

        if frame == 0 and fen_changed:
            self.logger.warning(f"Video don't start at the beginning")

        if self.continues_saving or fen_changed or frame == 0:
            move_number = self.move_number

            if self.save_per_frame:
                save_number = frame
            else:
                save_number = move_number

            if self.store_gt_boards and len(self.gt_board) != 0:
                cv2.imwrite(
                    f"{self.full_store_dir}/gt/{self.video_name}_{save_number:04d}.jpg",
                    self.gt_board,
                )

            if self.store_pred_boards:
                pred_board = chess.Board(self.fens[-1]["fen"])

                # Generate SVG and save as PNG
                svg_data = chess.svg.board(
                    board=pred_board,
                    size=400,
                    colors={"square light": "#f0d9b5", "square dark": "#b58863"},
                )

                # Save the SVG as PNG
                pred_img_path = f"{self.full_store_dir}/pred/{self.video_name}_{save_number:04d}.png"
                cairosvg.svg2png(
                    bytestring=svg_data.encode("utf-8"), write_to=pred_img_path
                )

            if self.store_masks and len(self.masked_img) != 0:
                cv2.imwrite(
                    f"{self.full_store_dir}/mask/{self.video_name}_{save_number:04d}.jpg",
                    self.masked_img,
                )
            if self.irl_board is not None:
                if self.store_irl_boards and len(self.irl_board) != 0:
                    cv2.imwrite(
                        f"{self.full_store_dir}/irl/{self.video_name}_{save_number:04d}.jpg",
                        self.irl_board,
                    )

            self.is_first_frame = False
            self.move_number += 1

            return succeed
        return True

    def extract_gt_board(self, img, gt_board_loc, irl_board_loc):
        top, bottom, left, right = gt_board_loc
        gt_board = img[top:bottom, left:right, :].copy()
        irl_top, irl_bottom, irl_left, irl_right = irl_board_loc
        irl_board = img[irl_top:irl_bottom, irl_left:irl_right, :].copy()

        # if not first image fill with black
        if self.first_frame_img is None:
            img[top:bottom, left:right, :] = 0
            return img, gt_board, irl_board

        # If first image fill with first frame
        img[top:bottom, left:right, :] = self.first_frame_img[top:bottom, left:right, :]
        return img, gt_board, irl_board


def process_single_video(dataset_key, debug=False):
    # Load id2label map from render dataset
    id2label_map = load_id2label_map(RENDER_DATASET_DIR)

    # Get video extraction settings
    video_name = DATASET_VIDEOS_SETTINGS[dataset_key]["video_id"]
    video_path = DOWNLOADED_VIDEOS_DIR / f"{video_name}.mp4"

    gt_board_loc = DATASET_VIDEOS_SETTINGS[dataset_key]["gt_board_loc"]
    irl_board_loc = DATASET_VIDEOS_SETTINGS[dataset_key]["irl_board_loc"]

    time_range = DATASET_VIDEOS_SETTINGS[dataset_key]["time_range"]
    step_sec = DATASET_VIDEOS_SETTINGS[dataset_key]["step_sec"]

    initial_state = DATASET_VIDEOS_SETTINGS[dataset_key].get("initial_state", None)

    continues_saving = DATASET_VIDEOS_SETTINGS[dataset_key].get(
        "continues_saving", False
    )

    print(
        f"Processing video: {video_name}, Time range: {time_range}, Step sec: {step_sec}, Initial state: {initial_state}"
    )

    # Instance ground truth predictor loading trained model
    gt_predictor = GTChessboardPredictor(
        id2label_map,
        mlruns_dir=str(MLRUNS_DIR),
        run_id=RENDER_MODEL_RUN_ID,
        initial_state=initial_state,
    )

    # Instance video board extractor
    vbe = VideoBoardExtractor(
        video_path=str(video_path),
        gt_board_loc=gt_board_loc,
        irl_board_loc=irl_board_loc,
        time_range=time_range,
        step_sec=step_sec,
        store_masks=False,
        store_gt_boards=True,
        store_gt_video=True,
        store_irl_boards=True,
        store_irl_video=True,
        store_pred_boards=True,
        store_pred_video=True,
        save_img_freq=True,
        at_first_gt_frame=True,
        purge_output_dir=True,
        store_dir=str(REAL_DATASET_DIR),
        gt_board_predictor=gt_predictor,
        save_per_frame=True if debug else False,
        continues_saving=continues_saving,
    )

    # Process video
    vbe.process_video()


if __name__ == "__main__":

    if not os.path.exists(DOWNLOADED_VIDEOS_DIR):
        os.makedirs(DOWNLOADED_VIDEOS_DIR)

    def download_dataset_videos():
        dataset_videos_ids = [v["video_id"] for v in DATASET_VIDEOS_SETTINGS.values()]
        for video_id in dataset_videos_ids:
            pull_video(video_id, DOWNLOADED_VIDEOS_DIR)

    def predict_game_videos_parallel(max_workers=2, debug=False):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            for dataset_key in DATASET_VIDEOS_SETTINGS.keys():
                futures.append(
                    executor.submit(process_single_video, dataset_key, debug=debug)
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing video: {e}")

    def predict_game_videos():
        dataset_keys = DATASET_VIDEOS_SETTINGS.keys()
        for idx, dataset_key in enumerate(dataset_keys):
            print(f"Processing video {idx+1}/{len(dataset_keys)}")
            process_single_video(dataset_key)

    # download_dataset_videos()
    predict_game_videos_parallel(max_workers=3, debug=False)
