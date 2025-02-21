def get_center_box(bbox):
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return (center_x, center_y)


def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])


def measure_distance(pos1, pos2):
    return ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)**0.5


def measure_xy_distance(pos1, pos2):
    return (pos1[0]-pos2[0], pos1[1]-pos2[1])
