import pickle
import cv2  # type: ignore
import numpy as np


class CameraMovementestimator():
    def __init__(self, frame):

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
