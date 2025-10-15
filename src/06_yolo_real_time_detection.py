import pathlib
import time
from collections import deque

from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

import cv2
import torch
import mlflow
import numpy as np
import chess
import chess.svg
import cairosvg
from loguru import logger
from ultralytics import YOLO

from model_maps import CPIECE2IDX as CLASS2ID

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent

YOLO_MODEL_DIR = ROOT_DIR / "models" / "yolo" / "finetuned" / "chess_yolov8s_finetuned"


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


@dataclass
class SquareCenter:
    row: int
    col: int
    center: tuple
    warped_center: tuple
    piece: str = "0"  # Default empty square
    prob: float = 1


class RTChessboardPredictor:
    def __init__(
        self,
        id2label_map,
        square_size=299,
        yolo_model_dir=YOLO_MODEL_DIR,
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

        self.yolo_model_dir = yolo_model_dir
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

        if self.best_model:
            model_path = str(YOLO_MODEL_DIR / "weights" / "best.pt")
        else:
            model_path = str(YOLO_MODEL_DIR / "weights" / "last.pt")

        self.model = YOLO(model_path)

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

    def detect_frames(self, frames):
        piece_detections = []

        for frame in frames:
            piece_dict = self.detect_frame(frame)
            piece_detections.append(piece_dict)

        return piece_detections

    def detect_frame(self, frame, device=None):

        # results = self.model.track(frame, persist=False)[0]
        results = self.model.predict(frame, verbose=False, imgsz=1024)[0]
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

            # Estimate base center of the piece, since the pieces are higher it
            # assume the base is at 1/3 of the height
            x1, y1, x2, y2 = result
            base_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

            if not object_cls_name in list(self.id2label_map.values()):
                continue

            pieces_dict[len(pieces_dict)] = {
                "bbox": result,
                "base_center": base_center,
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
                f"Class: {class_name}",
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # Draw base center
            base_center = piece_data["base_center"]
            cv2.circle(
                img=frame,
                center=(int(base_center[0]), int(base_center[1])),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )

        return frame

    def predict(
        self,
        gt_board_image: np.ndarray,
        pts_src: np.ndarray,
        pts_dst: np.ndarray,
        rotate_angle: int,
        device: torch.device,
        logger: logger,
        use_previous_moves: bool = False,
        verbose: bool = False,
    ):

        # print("Predicting...")
        # Predict without previous moves
        if not use_previous_moves:
            predicted_board_arr, _, img_detected = self.predict_gt_board(
                gt_board_image,
                pts_src=pts_src,
                pts_dst=pts_dst,
                rotate_angle=rotate_angle,
                device=device,
            )
            fen = self.arr2fen(predicted_board_arr)
            self.board = chess.Board(fen)
            self.prev_board = None
            return fen, True

        # Predict with previous moves
        predicted_board_arr, pred_probs, img_detected = self.predict_gt_board(
            gt_board_image,
            pts_src=pts_src,
            pts_dst=pts_dst,
            rotate_angle=rotate_angle,
            device=device,
        )

        logger.info(f"Predicted board array:\n{predicted_board_arr}")

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
        return (
            fen,
            final_valid_move,
            changed,
            final_min_loss,
            reject_prev_pred,
            img_detected,
        )

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

    def predict_gt_board(
        self,
        gt_board_img: np.ndarray,
        pts_src: np.ndarray,
        pts_dst: np.ndarray,
        rotate_angle: int,
        device: torch.device,
    ):

        print("Pts src:", pts_src)
        print("Pts dst:", pts_dst)

        # 1. Predict pieces in the original image
        pieces_dict = self.detect_frame(gt_board_img, device=device)

        # 2. Define inverse transfromation matrix
        M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)

        # 2. Predict center of each square
        rtl, rtr, rbr, rbl = pts_src
        square_size = int(rtr[0] - rtl[0]) / 8
        tl, tr, br, bl = pts_dst
        warped_square_size = int(tr[0] - tl[0]) / 8

        # 3. Get center of each square in original image
        square_centers = []
        for row in range(8):
            for col in range(8):
                x = (col + 0.5) * warped_square_size
                y = (row + 0.5) * warped_square_size

                warped_point = np.array([[x, y]], dtype="float32")
                original_point = cv2.perspectiveTransform(
                    np.array([warped_point]), M_inv
                )[0][0]

                square_center = SquareCenter(
                    row=row,
                    col=col,
                    warped_center=(x, y),
                    center=(int(original_point[0]), int(original_point[1])),
                )
                square_centers.append(square_center)

        # 4. Assign pieces to squares based on closest distance
        # Naive solution
        # for square in square_centers:
        #     min_dist = float("inf")
        #     assigned_piece = "0"  # Default empty
        #     assigned_prob = 0.0
        #     for _, piece_data in pieces_dict.items():
        #         piece_center = piece_data["base_center"]
        #         dist = np.linalg.norm(np.array(square.center) - np.array(piece_center))
        #         if dist < min_dist and dist < (self.square_size / 2):
        #             min_dist = dist
        #             assigned_piece = piece_data["class_name"]
        #             assigned_prob = piece_data["accuracy"]
        #     square.piece = assigned_piece
        #     square.prob = assigned_prob
        # Collect detections into a list
        detections = []
        for _, piece_data in pieces_dict.items():
            detections.append(
                {
                    "center": piece_data["base_center"],
                    "class_name": piece_data["class_name"],
                    "accuracy": piece_data["accuracy"],
                }
            )

        num_squares = len(square_centers)
        num_detections = len(detections)

        # Build cost matrix: (num_squares x num_detections)
        cost_matrix = np.zeros((num_squares, num_detections))
        for i, square in enumerate(square_centers):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(np.array(square.center) - np.array(det["center"]))
                cost_matrix[i, j] = dist

        # Solve assignment (Hungarian algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Assign results
        for i, j in zip(row_ind, col_ind):
            dist = cost_matrix[i, j]
            if dist < square_size / 2:  # enforce threshold
                square_centers[i].piece = detections[j]["class_name"]
                square_centers[i].prob = detections[j]["accuracy"]
            else:
                square_centers[i].piece = "0"
                square_centers[i].prob = 0.0

        # All other squares not matched remain empty
        for i, square in enumerate(square_centers):
            if square.piece is None:
                square.piece = "0"
                square.prob = 0.0

        # 5. Create board array and probability dictionary
        pred_board_arr = np.empty((8, 8), dtype=object)
        pred_probs = [[{} for _ in range(8)] for _ in range(8)]
        for square in square_centers:
            pred_board_arr[square.row, square.col] = square.piece

            # Store probability distribution for each square
            square_pred_probs = {p: 0.0 for p in self.id2label_map.values()}
            if square.piece == "0":
                square_pred_probs["0"] = 1.0
            else:
                square_pred_probs[square.piece] = square.prob
                square_pred_probs["0"] = 1.0 - square.prob

            pred_probs[square.row][square.col] = square_pred_probs

        # 6. Draw detected pieces and square centers
        img_detected = self.draw_bbox(gt_board_img.copy(), pieces_dict)

        # 7. rotate board and pred as needed
        if rotate_angle == 90:
            pred_board_arr = np.rot90(pred_board_arr, -1)
            pred_probs = [list(x) for x in zip(*pred_probs[::-1])]
            img_detected = cv2.rotate(img_detected, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == -90:
            pred_board_arr = np.rot90(pred_board_arr, 1)
            pred_probs = [list(x) for x in zip(*pred_probs)][::-1]
            img_detected = cv2.rotate(img_detected, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return pred_board_arr, pred_probs, img_detected


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
        self,
        board_predictor: RTChessboardPredictor,
        time_sec=2,
        stable_frames=3,
    ):

        # Logging
        self.logger = logger

        # Prediction
        self.board_predictor = board_predictor
        self.time_sec = time_sec

        # Detection stability
        self.prev_frame = None  # To check for stability
        self.motion_history = deque(maxlen=10)
        self.cover_threshold = 0.0001
        self.motion_threshold = 0.0001
        self.stable_frames_required = stable_frames
        self.stable_counter = 0
        self.reference_brightness = None
        self.brightness_threshold = 0.04  # Adjust as needed

        # Camera
        self.corners = []
        self.square_size = board_predictor.square_size
        self.display_board_size = 500

        # Save transformation matrix for yolo model
        self.pts_src = None
        self.pts_dst = None
        self.M = None

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
        )  # [(x_top-left, y_top-left), (x_top-right, y_top-right), (x_bottom-right, y_bottom-right), (x_bottom-left, y_bottom-left)]
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(frame, M, (board_size, board_size))

        # Save source and destination points and transformation matrix
        self.pts_src = pts_src
        self.pts_dst = pts_dst
        self.M = M

        # Ask user for rotation the first time
        if not hasattr(self, "rotation_code"):
            self.rotation_code = self.select_rotation(warped_img=warped)
        warped = self.rotate_img(warped, self.rotation_code)
        return warped

    def rotate_img(self, img, rotation_code):
        if rotation_code == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_code == 2:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

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

    def is_board_stable(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            self.reference_brightness = np.mean(gray)  # set baseline
            return False

        # motion check
        diff = cv2.absdiff(self.prev_frame, gray)  # Pixel-wise difference
        _, thresh = cv2.threshold(
            diff, 25, 255, cv2.THRESH_BINARY
        )  # Convert any diff > 25 to white
        motion_fraction = (
            np.count_nonzero(thresh) / thresh.size
        )  # Fraction of changed pixels
        self.motion_history.append(
            motion_fraction
        )  # Update fraction of changed pixels history (last N frames)
        avg_motion = np.mean(self.motion_history)

        motion_stable = avg_motion < self.motion_threshold

        # brightness check
        current_brightness = np.mean(gray)  # Current frame brightness
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
        img_detected = np.zeros((400, 400, 3), dtype=np.uint8)  # Blank image
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue

            warped = self.warp_board(frame)

            # Always show the raw camera feed
            cv2.imshow("Camera Feed", frame)

            # Check if board is stable
            if self.is_board_stable(warped):
                now = time.time()
                if now - last_time > self.time_sec:
                    # Run prediction
                    fen, *_, img_detected = self.board_predictor.predict(
                        frame,
                        pts_src=self.pts_src,
                        pts_dst=self.pts_dst,
                        rotate_angle=(
                            90
                            if self.rotation_code == 1
                            else -90 if self.rotation_code == 2 else 0
                        ),
                        device=device,
                        logger=self.logger,
                        use_previous_moves=True,
                    )
                    print("Predicted FEN:", fen)
                    last_time = now
                    detected_board_img = self.detected_board(fen)

            else:
                logger.info("Board not stable, waiting...")

            # Show the warped board and detected board
            cv2.imshow("Warped Chessboard", warped)
            cv2.imshow("Detected Board", img_detected)
            cv2.imshow("Predicted Board", detected_board_img)
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
        yolo_model_dir=YOLO_MODEL_DIR,
        from_beginning=True,
        best_model=True,
    )
    rtc = RealTimeChessCamera(board_predictor=board_predictor, time_sec=2)
    rtc.run()
