from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    def __init__(self):
        pass

    def get_cluster_model(self, image):
        image2 = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image2)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

        top_player = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_cluster_model(top_player)
        labels = kmeans.labels_
        clustered_image = labels.reshape(
            top_player.shape[0], top_player.shape[1])
        corner_cluster = [clustered_image[0, 0], clustered_image[0, -1],
                          clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

    def assign_color(self, frame, player_detect):
        player_color = []
        for _, player_detect in player_detect.items():
            bbox = player_detect['bbox']
            player_color = self.get_player_color(frame, bbox)
