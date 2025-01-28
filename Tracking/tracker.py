from utils import get_bbox_width, get_center_box
from ultralytics import YOLO  # type: ignore
import supervision as sv  # type: ignore
import os
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
                frames[i:i+n_frames], conf=0.1)
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

    def draw_annot(self, video_frame, tracks):
        output_frames = []
        for frame_N, frame in enumerate(video_frame):
            frame = frame.copy()
            player_dictio = tracks["players"][frame_N]
            ball_dictio = tracks["ball"][frame_N]
            ref_dictio = tracks["referees"][frame_N]

            for track_id, player in player_dictio.items():
                frame = self.draw_ellipse(
                    frame, player["box_detect"], (255, 0, 0), track_id)

            for track_id, ball in ball_dictio.items():
                frame = self.draw_ellipse(
                    frame, ball["box_detect"], (0, 0, 255), track_id)

            for track_id, ref in ref_dictio.items():
                frame = self.draw_ellipse(
                    frame, ref["box_detect"], (255, 255, 255), track_id)

            output_frames.append(frame)
        return output_frames

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])
        center_x = get_center_box(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, center=(center_x, y2), axes=(int(width), int(0.35 * width)), angle=0.0,
                    startAngle=-45, endAngle=235, color=color, thickness=3, lineType=cv2.LINE_4)  # Provide the minor and major axes of the ellipse
        return frame
