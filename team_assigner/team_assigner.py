from sklearn.cluster import KMeans
import numpy as np
import cv2  # type: ignore


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.team_player = {}

    def get_cluster_model(self, image):
        image2 = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans = kmeans.fit(image2)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Ensure the bounding box is valid
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError("Invalid bounding box coordinates")

        image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        top_player = image[0:int(image.shape[0]/2), :]

        pixels = top_player.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
        kmeans.fit(pixels)

        # Determine which cluster corresponds to the player
        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_player.shape[0], top_player.shape[1])
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1],
                          clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        # Get the dominant color of the player
        player_color = kmeans.cluster_centers_[player_cluster]
        player_color = cv2.cvtColor(
            np.uint8([[player_color]]), cv2.COLOR_LAB2BGR)[0][0]

        return player_color

    def get_team_player(self, frame, player_bbox, player_id):
        if player_id in self.team_player:
            return self.team_player[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        if player_id == 91:
            team_id = 1

        self.team_player[player_id] = team_id
        return team_id

    def assign_color(self, frame, player_detect):
        player_colors = []
        for _, player_detect in player_detect.items():
            bbox = player_detect['box_detect']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++",
                        n_init=10)
        kmeans = kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
