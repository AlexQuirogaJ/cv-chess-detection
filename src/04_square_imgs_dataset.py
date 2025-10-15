import pathlib
import cv2
import gradio as gr
import numpy as np
from typing import List

current_dir = pathlib.Path.cwd()
ROOT_DIR = current_dir.parent
# REAL_DATASET_NAME = "real_chess_dataset"
REAL_DATASET_NAME = "recorded_dataset"
REAL_DATASET_DIR = ROOT_DIR / "data" / REAL_DATASET_NAME

RENDER_IMG_WIDTH = 800
SQUARE_IMG_WIDTH = 512


def get_video_ids() -> List[str]:
    """Get the video ids from the real dataset directory"""
    video_ids = [
        d.name
        for d in REAL_DATASET_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return video_ids


def get_image(video_id: str, move_number: int) -> np.ndarray:
    """Get the image from the real dataset directory"""
    if not video_id:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    irl_img_path = (
        REAL_DATASET_DIR / f"{video_id}" / "irl" / f"{video_id}_{move_number:04}.jpg"
    )

    if not irl_img_path.exists():
        return np.zeros((100, 100, 3), dtype=np.uint8)

    irl_img = cv2.imread(str(irl_img_path))
    irl_rgb = cv2.cvtColor(irl_img, cv2.COLOR_BGR2RGB)
    h, w = irl_rgb.shape[:2]
    irl_rgb_scaled = cv2.resize(
        irl_rgb, (RENDER_IMG_WIDTH, int(h * RENDER_IMG_WIDTH / w))
    )
    return irl_rgb_scaled


def get_max_moves(video_id: str) -> int:
    """Get the maximum number of moves for a video"""
    if not video_id:
        return 1
    irl_dir = REAL_DATASET_DIR / video_id / "irl"
    if not irl_dir.exists():
        return 1
    return len(list(irl_dir.glob("*.jpg")))


def warp_image(img: np.ndarray, pts: List[List[int]]) -> np.ndarray:
    """Warp the image based on the selected points"""
    if img is None or not pts or len(pts) != 4:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    input_pts = np.float32(pts)
    output_pts = np.float32(
        [
            [0, 0],
            [SQUARE_IMG_WIDTH, 0],
            [SQUARE_IMG_WIDTH, SQUARE_IMG_WIDTH],
            [0, SQUARE_IMG_WIDTH],
        ]
    )

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply the perspective transformation to the image
    warped_img = cv2.warpPerspective(img, M, (512, 512), flags=cv2.INTER_LINEAR)

    return warped_img


def save_all_squares(
    video_id: str, pts: List[List[int]], rotate_cw: bool, rotate_ccw: bool
) -> str:
    """Save all the squared images for a video"""
    if not video_id or not pts or len(pts) != 4:
        return "Invalid video ID or points"

    irl_dir = REAL_DATASET_DIR / video_id / "irl"
    squared_dir = REAL_DATASET_DIR / video_id / "squared"
    squared_dir.mkdir(exist_ok=True)

    input_pts = np.float32(pts)
    output_pts = np.float32(
        [
            [0, 0],
            [SQUARE_IMG_WIDTH, 0],
            [SQUARE_IMG_WIDTH, SQUARE_IMG_WIDTH],
            [0, SQUARE_IMG_WIDTH],
        ]
    )
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    for img_path in irl_dir.glob("*.jpg"):
        irl_img = cv2.imread(str(img_path))
        irl_rgb = cv2.cvtColor(irl_img, cv2.COLOR_BGR2RGB)
        h, w = irl_rgb.shape[:2]
        irl_rgb_scaled = cv2.resize(
            irl_rgb, (RENDER_IMG_WIDTH, int(h * RENDER_IMG_WIDTH / w))
        )
        warped_img = cv2.warpPerspective(
            irl_rgb_scaled,
            M,
            (SQUARE_IMG_WIDTH, SQUARE_IMG_WIDTH),
            flags=cv2.INTER_LINEAR,
        )
        save_path = squared_dir / img_path.name
        # convert to BGR before saving
        warped_img_bgr = cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR)

        if rotate_cw:
            warped_img_bgr = cv2.rotate(warped_img_bgr, cv2.ROTATE_90_CLOCKWISE)
        if rotate_ccw:
            warped_img_bgr = cv2.rotate(warped_img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(str(save_path), warped_img_bgr)

    return f"Saved squared images to {squared_dir}"


with gr.Blocks() as demo:
    # UI and states
    gr.Markdown("# Chessboard Squarer")
    selected_points = gr.State([])  # Hidden state for selected points (in server)
    with gr.Row():
        with gr.Column(scale=1):
            video_id_dropdown = gr.Dropdown(
                choices=get_video_ids(), label="Video ID", value=None
            )
            max_moves = gr.State(1)  # Hidden state for max moves
            move_number_slider = gr.Slider(
                minimum=1, maximum=max_moves.value, step=1, label="Move Number"
            )
            with gr.Row():
                # Rotate 90º clockwise
                # rotate_cw_button = gr.Button("⤵️", scale=1)
                rotate_cw_checkbox = gr.Checkbox(label="⤵️", value=False)
                # Rotate 90º counter-clockwise
                # rotate_ccw_button = gr.Button("⤴️", scale=1)
                rotate_ccw_checkbox = gr.Checkbox(label="⤴️", value=False)
            process_button = gr.Button("Create Squared Images")
            status_textbox = gr.Textbox(label="Status")
        with gr.Column(scale=2):
            input_image = gr.Image(label="Select 4 corners")
            reset_button = gr.Button("Reset Selection")
            output_image = gr.Image(label="Warped Image")

    # Callback select video > update move slider
    def update_move_slider(video_id):
        max_m = get_max_moves(video_id)
        return gr.Slider(minimum=1, maximum=max_m, step=1, label="Move Number", value=1)

    video_id_dropdown.change(
        fn=update_move_slider, inputs=video_id_dropdown, outputs=move_number_slider
    )

    # Callback select video or move number > update image
    def update_image(video_id, move_number):
        img = get_image(video_id, move_number)
        # selected_points.value = []  # Reset points on new image
        return img, None, []

    move_number_slider.change(
        fn=update_image,
        inputs=[video_id_dropdown, move_number_slider],
        outputs=[input_image, output_image, selected_points],
    )
    video_id_dropdown.change(
        fn=update_image,
        inputs=[video_id_dropdown, move_number_slider],
        outputs=[input_image, output_image, selected_points],
    )

    # Callback reset selection
    reset_button.click(
        fn=update_image,
        inputs=[video_id_dropdown, move_number_slider],
        outputs=[input_image, output_image, selected_points],
    )

    # Callback select points on image
    def handle_select(img, evt: gr.SelectData, selected_points: list):
        if img is None:
            return None, None

        # Append new point (x, y)
        new_points = selected_points + [evt.index]

        # Draw points on image
        img_with_points = img.copy()
        for pt in new_points:
            cv2.circle(img_with_points, tuple(pt), 5, (255, 0, 0), -1)

        if len(new_points) == 4:
            warped = warp_image(img, new_points)
            # Don't reset points here, allow saving
            return (img_with_points, warped, new_points)

        return img_with_points, None, new_points

    input_image.select(
        handle_select,
        [input_image, selected_points],
        [input_image, output_image, selected_points],
    )

    process_button.click(
        fn=save_all_squares,
        inputs=[
            video_id_dropdown,
            selected_points,
            rotate_cw_checkbox,
            rotate_ccw_checkbox,
        ],
        outputs=status_textbox,
    )

if __name__ == "__main__":
    demo.launch()
