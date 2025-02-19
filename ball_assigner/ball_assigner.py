from utils import get_center_box, measure_distance
import sys
sys.path.append('../')


class BallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_box):
        ball_pos = get_center_box(ball_box)
        print(f"Ball position: {ball_pos}")  # Debugging: Check ball position

        min_distance = 99999999
        assigned_player = -1

        for player_id, player in players.items():
            player_pos = player['box_detect']

            # Debugging: Check the type of player_pos
            if not isinstance(player_pos, (list, tuple)):
                print(
                    f"Warning: player_pos for player {player_id} is not a list or tuple: {player_pos}")
                continue

            # Debugging: Check the length of player_pos
            if len(player_pos) < 4:
                print(
                    f"Warning: player_pos for player {player_id} has insufficient elements: {player_pos}")
                continue

            # Debugging: Print player_pos and ball_pos
            print(
                f"Player {player_id} position: {player_pos}, Ball position: {ball_pos}")

            l_distance = measure_distance(
                (player_pos[0], player_pos[-1]), ball_pos)
            r_distance = measure_distance(
                (player_pos[2], player_pos[-1]), ball_pos)
            distance = min(l_distance, r_distance)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player
