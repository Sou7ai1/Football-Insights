def get_center_box(bbox):
    return int((bbox[0] + bbox[2]) / 2)


def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])


def measure_distance(pos1, pos2):
    return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5
