import pathlib
import time
from collections import deque

import cv2
import torch
import mlflow
import numpy as np
import chess
import chess.svg
import cairosvg
from loguru import logger

from model_maps import CPIECE2IDX as CLASS2ID

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

RENDER_DATASET_NAME = "render_chess_dataset"
RENDER_DATASET_DIR = ROOT_DIR / "data" / RENDER_DATASET_NAME

MLRUNS_DIR = ROOT_DIR / "data" / "mlruns"
# MODEL_RUN_ID = "0c2d1755485045b28e5048e624b5d6d4"
MODEL_RUN_ID = "4756c5c248a248688ad31f0362089ca1"

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

MLFLOW_EXPERIMENT_NAME = "RecordedChessModel"
MLRUNS_DIR = ROOT_DIR / "data" / "mlruns"


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


class RTChessboardPredictor:
    def __init__(
        self,
        id2label_map,
        square_size=299,
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


class RealTimeChessCamera:
    """Class to handle real-time chessboard detection and prediction from a camera feed.

    1. Connects to a USB camera using OpenCV.
    2. Lets the user select the four corners of the chessboard in the camera feed.
    3. Warps the selected chessboard area to a top-down view.
    4. Periodically captures the warped chessboard image and uses a provided
       chessboard predictor to predict the board state.
    5. Displays the camera feed and the warped chessboard in real-time.
    """

    def __init__(
        self, board_predictor: RTChessboardPredictor, time_sec=2, stable_frames=3
    ):

        # Logging
        self.logger = logger

        # Prediction
        self.board_predictor = board_predictor
        self.time_sec = time_sec

        # Detection stability
        self.prev_frame = None  # To check for stability
        self.motion_history = deque(maxlen=10)
        self.cover_threshold = 0.0002  # 0.0001
        self.motion_threshold = 0.0002  # 0.0001
        self.stable_frames_required = stable_frames
        self.stable_counter = 0
        self.reference_brightness = None
        self.brightness_threshold = 0.04  # Adjust as needed

        # Camera
        self.corners = []
        self.square_size = board_predictor.square_size
        self.display_board_size = 500

    def select_corners(self):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
                self.corners.append((x, y))

        print("Select the 4 corners of the chessboard (clockwise from top-left).")
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue
            display = frame.copy()
            for corner in self.corners:
                cv2.circle(
                    img=display,
                    center=corner,
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1,
                )
            cv2.imshow("Select Corners", display)
            cv2.setMouseCallback("Select Corners", mouse_callback)
            if len(self.corners) == 4:
                break
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        cv2.destroyWindow("Select Corners")

    def warp_board(self, frame):
        pts_src = np.array(self.corners, dtype="float32")
        # board_size = self.square_size * 8
        board_size = self.display_board_size
        pts_dst = np.array(
            [
                [0, 0],
                [board_size - 1, 0],
                [board_size - 1, board_size - 1],
                [0, board_size - 1],
            ],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, M, (board_size, board_size))

        # Ask user for rotation the first time
        if not hasattr(self, "rotation_code"):
            self.rotation_code = self.select_rotation(warped_img=warped)
        if self.rotation_code == 1:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_code == 2:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return warped

    def detected_board(self, fen):
        pred_board = chess.Board(fen)

        # Generate SVG and save as PNG
        svg_data = chess.svg.board(
            board=pred_board,
            size=400,
            colors={"square light": "#f0d9b5", "square dark": "#b58863"},
        )

        # Convert SVG bytes to PNG bytes in memory
        png_bytes = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))

        # Convert PNG bytes to NumPy array (OpenCV image)
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img

    def select_camera_cli(self):
        # Print available cameras
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                camera_info = cap.getBackendName()
                print(f"Idx: {index} - {camera_info}")
                arr.append(index)
            cap.release()
            index += 1
        print("Available cameras:", arr)
        camera_id = int(input("Select camera ID (default 0): ") or 0)

        return camera_id

    def select_camera_gui(self):
        # Scan for available cameras and capture one frame from each
        previews = []
        indices = []
        max_cams = 8  # Limit for speed
        for idx in range(max_cams):
            cap = cv2.VideoCapture(idx)

            # End checking if no camera is found
            if not cap.read()[0]:
                cap.release()
                break

            start_time = time.time()
            while time.time() - start_time < 2:  # 1 second timeout
                cap.read()
            ret, frame = cap.read()
            if ret:
                preview = cv2.resize(frame, (320, 240))
                previews.append(preview)
                indices.append(idx)
            cap.release()
        if not previews:
            raise RuntimeError("No cameras found.")

        # Create a grid image
        grid_cols = 2
        grid_rows = (len(previews) + 1) // grid_cols
        grid_img = np.zeros((grid_rows * 240, grid_cols * 320, 3), dtype=np.uint8)
        positions = []
        for i, preview in enumerate(previews):
            row = i // grid_cols
            col = i % grid_cols
            y, x = row * 240, col * 320
            grid_img[y : y + 240, x : x + 320] = preview
            cv2.putText(
                grid_img,
                f"Cam {indices[i]}",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),  # White border
                4,  # Thickness for border
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                grid_img,
                f"Cam {indices[i]}",
                (x + 10, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),  # Black text
                2,  # Thickness for text
                lineType=cv2.LINE_AA,
            )
            positions.append((x, y, x + 320, y + 240, indices[i]))

        selected = [-1]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for x1, y1, x2, y2, idx in positions:
                    if x1 <= x < x2 and y1 <= y < y2:
                        selected[0] = idx
                        cv2.destroyWindow("Select Camera")
                        break

        cv2.namedWindow("Select Camera")
        cv2.setMouseCallback("Select Camera", mouse_callback)
        while selected[0] == -1:
            cv2.imshow("Select Camera", grid_img)
            if cv2.waitKey(50) & 0xFF == 27:
                break
        cv2.destroyWindow("Select Camera")
        return selected[0] if selected[0] != -1 else indices[0]

    def select_rotation(self, warped_img=None):
        """
        UI to select rotation: clockwise, counterclockwise, or no rotation,
        showing the warped image with each rotation option.
        Returns: rotation_code (int): 0=no rotation, 1=90° CW, 2=90° CCW
        """
        options = ["Don't rotate", "Rotate 90 CW", "Rotate 90 CCW"]
        selected = [0]
        img_h, img_w = 300, 300

        # Prepare images for each rotation
        if warped_img is not None:
            warped_resized = cv2.resize(warped_img, (img_w, img_h))
            warped_cw = cv2.rotate(warped_resized, cv2.ROTATE_90_CLOCKWISE)
            warped_ccw = cv2.rotate(warped_resized, cv2.ROTATE_90_COUNTERCLOCKWISE)
            warped_options = [warped_resized, warped_cw, warped_ccw]
        else:
            warped_options = [
                np.zeros((img_h, img_w, 3), dtype=np.uint8) for _ in range(3)
            ]

        def draw_options():
            # Create a single image with all three options side by side
            gap = 10
            total_w = img_w * 3 + gap * 4
            total_h = img_h + 80
            display = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            for i, img in enumerate(warped_options):
                x = gap + i * (img_w + gap)
                display[40 : 40 + img_h, x : x + img_w] = img
                color = (0, 255, 0) if i == selected[0] else (255, 255, 255)
                cv2.rectangle(display, (x, 40), (x + img_w, 40 + img_h), color, 3)
                cv2.putText(
                    display,
                    f"{i+1}. {options[i]}",
                    (x + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            return display

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for i in range(3):
                    x0 = 10 + i * (img_w + 10)
                    x1 = x0 + img_w
                    if x0 <= x < x1 and 40 <= y < 40 + img_h:
                        selected[0] = i

        cv2.namedWindow("Select Rotation")
        cv2.setMouseCallback("Select Rotation", mouse_callback)
        while True:
            display = draw_options()
            cv2.imshow("Select Rotation", display)
            key = cv2.waitKey(50)
            if key in [49, 50, 51]:  # 1, 2, 3 keys
                selected[0] = key - 49
            if key == 13 or key == 10:  # Enter
                break
            if key == 27:  # ESC
                selected[0] = 0
                break
        cv2.destroyWindow("Select Rotation")
        return selected[0]

    # def is_board_stable(self, frame, threshold=1):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #     if self.prev_frame is None:
    #         self.prev_frame = gray
    #         return False

    #     diff = cv2.absdiff(self.prev_frame, gray)
    #     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    #     motion_level = np.sum(thresh) / 255

    #     # Determine if the board is stable
    #     stable = motion_level < threshold
    #     self.prev_frame = gray

    #     print(f"Motion level: {motion_level}, Stable: {stable}")

    #     return stable

    # def is_board_stable(self, frame):
    #     # Convert to grayscale and blur
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #     if self.prev_frame is None:
    #         self.prev_frame = gray
    #         return False

    #     # Motion detection
    #     diff = cv2.absdiff(self.prev_frame, gray)
    #     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    #     motion_level = np.sum(thresh) / 255
    #     motion_stable = motion_level < self.motion_threshold

    #     # Cover detection: look at overall brightness difference
    #     mean_diff = np.abs(np.mean(gray) - np.mean(self.prev_frame))
    #     cover_detected = mean_diff > self.cover_threshold

    #     # Update prev_frame for next iteration
    #     self.prev_frame = gray

    #     # Stable if no significant motion and no cover detected
    #     stable = motion_stable and not cover_detected

    #     print(f"Motion level: {motion_level}, Mean diff: {mean_diff}, Stable: {stable}")
    #     return stable

    # def is_board_stable(self, frame):
    #     # Convert to grayscale + blur
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)

    #     if self.prev_frame is None:
    #         self.prev_frame = gray
    #         return False

    #     # Motion
    #     # - Computes absolute diff per-pixel
    #     diff = cv2.absdiff(self.prev_frame, gray)
    #     # Convert to binary image if pixel change > 25
    #     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    #     # Fraction of pixels with motion
    #     motion_fraction = np.count_nonzero(thresh) / thresh.size  # < 1.0
    #     self.motion_history.append(motion_fraction)

    #     # Rolling average of motion
    #     avg_motion = np.mean(self.motion_history)

    #     # Large motion area → likely covering the board
    #     cover_detected = avg_motion > self.cover_threshold

    #     # Stable if motion below threshold and not occluded
    #     motion_stable = avg_motion < self.motion_threshold
    #     stable = motion_stable and not cover_detected

    #     # Require multiple consecutive stable frames
    #     if stable:
    #         self.stable_counter += 1
    #     else:
    #         self.stable_counter = 0

    #     self.prev_frame = gray

    #     is_really_stable = self.stable_counter >= self.stable_frames_required

    #     print(
    #         f"Motion fraction: {motion_fraction:.4f}, "
    #         f"Avg motion: {avg_motion:.4f}, "
    #         f"Cover: {cover_detected}, "
    #         f"Stable counter: {self.stable_counter}, "
    #         f"Board stable: {is_really_stable}"
    #     )

    #     return is_really_stable

    def is_board_stable(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            self.reference_brightness = np.mean(gray)  # set baseline
            return False

        # motion check
        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_fraction = np.count_nonzero(thresh) / thresh.size
        self.motion_history.append(motion_fraction)
        avg_motion = np.mean(self.motion_history)

        motion_stable = avg_motion < self.motion_threshold

        # brightness check
        current_brightness = np.mean(gray)
        brightness_diff = (
            abs(current_brightness - self.reference_brightness)
            / self.reference_brightness
        )
        occluded = brightness_diff > self.brightness_threshold

        stable = motion_stable and not occluded

        if stable:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        self.prev_frame = gray
        is_really_stable = self.stable_counter >= self.stable_frames_required

        print(
            f"Motion: {motion_fraction:.4f}, Avg: {avg_motion:.4f}, "
            f"Brightness diff: {brightness_diff:.4f}, "
            f"Occluded: {occluded}, Stable counter: {self.stable_counter}, "
            f"Board stable: {is_really_stable}"
        )

        # Update reference brightness only if really stable
        if is_really_stable:
            self.reference_brightness = current_brightness

        return is_really_stable

    def run(self):

        # Select camera
        camera_id = self.select_camera_gui()
        # camera_id = self.select_camera_cli()
        self.capture = cv2.VideoCapture(camera_id)

        # Select device for prediction model
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple GPU
        else:
            device = torch.device("cpu")

        self.logger.info(f"Using device: {device}")
        self.board_predictor.model.to(device)

        # Select corners of the chessboard
        self.select_corners()

        # Predict
        last_time = 0
        detected_board_img = np.zeros((400, 400, 3), dtype=np.uint8)  # Blank image
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue

            warped = self.warp_board(frame)

            # Always show the raw camera feed
            cv2.imshow("Camera Feed", frame)

            # Only update detected_board_img if stable
            if self.is_board_stable(warped):
                now = time.time()
                if now - last_time > self.time_sec:
                    # Run prediction
                    fen, *_ = self.board_predictor.predict(
                        warped, device, logger=self.logger, use_previous_moves=True
                    )
                    print("Predicted FEN:", fen)
                    last_time = now
                    detected_board_img = self.detected_board(fen)
            else:
                # Avoid detection when moving the pieces
                logger.info("Board not stable, waiting...")

            # Show the warped board and detected board
            cv2.imshow("Warped Chessboard", warped)
            cv2.imshow("Detected Board", detected_board_img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        self.capture.release()
        cv2.destroyAllWindows()


# mlflow.set_tracking_uri("file://" + str(MLRUNS_DIR))
# model = mlflow.pytorch.load_model("runs:/" + RUN_ID + "/best_model")


if __name__ == "__main__":

    board_predictor = RTChessboardPredictor(
        id2label_map={v: k for k, v in CLASS2ID.items()},
        square_size=299,
        run_id=RUN_ID,
        mlruns_dir=str(MLRUNS_DIR),
        from_beginning=True,
        best_model=True,
    )
    rtc = RealTimeChessCamera(board_predictor=board_predictor, time_sec=2)
    rtc.run()
