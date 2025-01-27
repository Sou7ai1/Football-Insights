def get_center_box(bbox):
    return int((bbox[0] + bbox[2]) / 2)

def get_bbox_width(bbox):
    return int(bbox[2] - bbox[0])