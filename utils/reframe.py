import cv2  # type: ignore
import os

if not os.path.exists("Videos_Input/08fd33_4.mp4"):
    print("Error: Input video file does not exist.")

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
    return frames


def save_video(output_video, output_path):
    if not output_video or output_video[0] is None:
        print("Error: Output video is empty or contains None values.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 24, (
        output_video[0].shape[1], output_video[0].shape[0]))  # Path,frame/s,imgsize
    for frame in output_video:
        out.write(frame)

    out.release()


