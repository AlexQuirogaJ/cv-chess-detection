import os
import cv2


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def read_video(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the video
    if original_fps <= 0:
        raise ValueError("Invalid FPS value from video file.")

    if fps <= 0:
        raise ValueError("FPS must be a positive integer.")

    frame_interval = int(original_fps // fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(0, total_frames, frame_interval):
        # Move to frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:  # If no frame is returned, we reached the end of the video
            break

        frames.append(frame)
    cap.release()  # Release the video capture object
    return frames


def save_video(output_video_frames, output_video_path, output_fps=24):
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
        output_fps,  # 24 FPS
        (
            output_video_frames[0].shape[1],
            output_video_frames[0].shape[0],
        ),  # (width, height)
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
