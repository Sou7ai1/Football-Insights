from utils import read_video, save_video
from Tracking import Tracker
import cv2  # type: ignore
import numpy as np
from team_assigner import TeamAssigner
from ball_assigner import BallAssigner
from camera_movement_estimator import CameraMovementestimator


def main():
    frames = read_video("Videos_Input/08fd33_4.mp4")

    tracker = Tracker('modelsV11X/best.pt')

    tracks = tracker.get_object(
        frames, Read_file=True, Rpath="stubs/track_stub.pkl")

    Camera_Movement_estimator = CameraMovementestimator(frames[0])
    Camera_Movement_estimator_frames = Camera_Movement_estimator.get_camera_movement(
        frames, read_from=True, stub_path="stubs/camera_movement.pkl")

    tracks["ball"] = tracker.ball_interpol(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_team_player(
                frames[frame_num], track['box_detect'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = BallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_frame = tracks['ball'][frame_num][1]['box_detect']
        assigned_player = player_assigner.assign_ball(
            player_track, ball_frame)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Crop Image To Visualize Better
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['box_detect']
    #     frame = frames[0]

    #     cropped_image = frame[int(bbox[1]):int(
    #         bbox[3]), int(bbox[0]):int(bbox[2])]
    #     cv2.imwrite(f"Output_Images/player_{track_id}.jpg", cropped_image)

    #     break

    output = "Output_Videos/output_video.avi"
    output_video_frames = tracker.draw_annot(frames, tracks, team_ball_control)
    output_video_frames = Camera_Movement_estimator.draw_camera_movement(
        output_video_frames, Camera_Movement_estimator_frames)

    save_video(output_video_frames, output)


if __name__ == "__main__":
    main()
