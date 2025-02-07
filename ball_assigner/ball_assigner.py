from utils import get_center_box, measure_distance
import sys
sys.path.append('../')


class BallAssigner:
    def init(self):
        self.max_player_ball_distance = 70

    def assign_ball(self, players, ball_box):
        ball_pos = get_center_box(ball_box)

        min_distance = 9000000000000
        assigned_player = -1

        for player_id, player in players.items():
            player_pos = player['box_detect']
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
