from utils import get_bbox_width, get_center_box
from ultralytics import YOLO  # type: ignore
import supervision as sv  # type: ignore
import os
import numpy as np
import pandas as pd
import pickle
import cv2  # type: ignore
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):  # Frame by frame detection
        n_frames = 20
        results = []
        for i in range(0, len(frames), n_frames):
            detections_n_frames = self.model.predict(
                frames[i:i+n_frames], conf=0.1, verbose=False)  # Disabling verbose mode
            results += detections_n_frames
        return results

    def get_object(self, frames, Read_file=False, Rpath=None):
        if Read_file and Rpath and os.path.exists(Rpath):
            with open(Rpath, 'rb') as file:
                track = pickle.load(file)

        detections = self.detect_frames(frames)

        # CHECK AFTER TO KEEP Goalkeeper
        track = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            # Inverse dictionary
            cls_names_inverse = {v: k for k, v in cls_names.items()}
            detection_sv = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_sv.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_sv.class_id[object_ind] = cls_names_inverse['player']

            detect_tracks = self.tracker.update_with_detections(
                detection_sv)

            track["players"].append({})
            track["referees"].append({})
            track["ball"].append({})

            for frame_detection in detect_tracks:
                box_detect = frame_detection[0].tolist()
                clas_id = frame_detection[3]
                track_id = frame_detection[4]

                if clas_id == cls_names_inverse['player']:
                    track["players"][frame_num][track_id] = {
                        "box_detect": box_detect}

                if clas_id == cls_names_inverse['referee']:
                    track["referees"][frame_num][track_id] = {
                        "box_detect": box_detect}

            for frame_detection in detection_sv:
                box_detect = frame_detection[0].tolist()
                clas_id = frame_detection[3]
                if clas_id == cls_names_inverse['ball']:
                    track["ball"][frame_num][1] = {
                        "box_detect": box_detect}

        if Read_file:
            with open(Rpath, 'wb') as file:
                pickle.dump(track, file)

        return track

    def draw_annot(self, video_frame, tracks, team_ball_control):
        output_frames = []
        for frame_N, frame in enumerate(video_frame):
            frame = frame.copy()
            player_dictio = tracks["players"][frame_N]
            ball_dictio = tracks["ball"][frame_N]
            ref_dictio = tracks["referees"][frame_N]

            for track_id, player in player_dictio.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(
                    frame, player["box_detect"], color, track_id)
                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame, player["box_detect"], (0, 0, 255))

            for _, ref in ref_dictio.items():
                frame = self.draw_ellipse(
                    frame, ref["box_detect"], (0, 255, 255))

            for track_id, ball in ball_dictio.items():
                frame = self.draw_triangle(
                    frame, ball["box_detect"], (0, 255, 0))

            frame = self.draw_team_ball_control(
                frame, frame_N, team_ball_control)

            output_frames.append(frame)
        return output_frames

    def draw_team_ball_control(self, frame, frame_N, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 950),
                      (255, 255, 255), cv2.FILLED)
        alpha = 0.4  # for transparency
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_frame = team_ball_control[:frame_N+1]
        team_A_frames = team_ball_control_frame[team_ball_control_frame == 1].shape[0]
        team_B_frames = team_ball_control_frame[team_ball_control_frame == 2].shape[0]

        team_A = team_A_frames/(team_A_frames+team_B_frames)
        team_B = team_B_frames/(team_A_frames+team_B_frames)

        cv2.putText(frame, f"Team A Ball Control: {team_A*100:.2f}%",
                    (1400, 890), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team B Ball Control: {team_B*100:.2f}%",
                    (1400, 945), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        center_x, _ = get_center_box(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(center_x, y2), axes=(int(width), int(0.35 * width)), angle=0.0,
                    startAngle=-45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)  # Provide the minor and major axes of the ellipse

        rectangle_width = 40
        rectangle_height = 10
        x1_rectangle = center_x - rectangle_width // 2
        x2_rectangle = center_x + rectangle_width // 2
        y1_rectangle = (y2 - rectangle_height // 2) + 15
        y2_rectangle = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rectangle), int(y1_rectangle + 30)),
                          (int(x2_rectangle), int(y2_rectangle)), (255, 255, 255), cv2.FILLED)

            text = f"{track_id}"
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            x1_text = x1_rectangle + (rectangle_width - text_size[0]) // 2
            y1_text = y1_rectangle + 30 + \
                (rectangle_height + text_size[1]) // 2

            cv2.putText(frame, text, (int(x1_text+5), int(y1_text-5)),
                        font, font_scale, (0, 0, 0), thickness)
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_box(bbox)

        triangle = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frame

    def ball_interpol(self, ball_positions):
        ball_positions = [x.get(1, {}).get('box_detect', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"box_detect": x}}
                          for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
