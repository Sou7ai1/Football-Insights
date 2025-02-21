from utils import measure_xy_distance, measure_distance
from utils import measure_distance
import pickle
import cv2  # type: ignore
import numpy as np
import sys
import os
sys.path.append('..')


class CameraMovementestimator():
    def __init__(self, frame):

        self.min_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS |
                      cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        frame_one = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(frame_one)
        mask_features[:, 900:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,  # Search Size of the Features
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from=False, stub_path=None):

        if read_from and stub_path is not None and os.path.exists(stub_path): 
            with open(read_from, 'rb') as f:
                return pickle.load(f)
        
        camera_movement = [[0, 0]*len(frames)]
        old_color = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_color, **self.features)

        for frame_N in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_N], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_color, frame_gray, old_features, None, **self.lk_params)
            max_dis = 0
            camera_movement_x, camera_movement_y = 0, 0
            for i, (new, old) in enumerate(new_features, old_features):
                new_features_p = new.ravel()
                old_features_p = old.ravel()
                distance = measure_distance(new_features_p, old_features_p)
                if distance > max_dis:
                    max_dis = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_p, new_features_p)
                    
            if max_dis < self.min_distance:
                camera_movement[frame_N] = [
                    camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)
                
            old_color = frame_gray.copy()
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement)
            
        return camera_movement
