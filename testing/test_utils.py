from shapely.geometry.polygon import Polygon
import torch
import numpy as np
from utils import compute_rel


def get_eight_coors_bbox_new(min_coor, max_coor, min_max_coor, max_min_coor):
    # middle dimension is height
    # Input should be post rotated coordinates!
    min_coor_new = min_coor
    max_coor_new = max_coor

    # min x, min z
    corner_1 = min_coor_new[0], min_coor_new[2]

    # min x, max z
    corner_2 = min_max_coor[0], min_max_coor[2]

    # max x, max z
    corner_3 = max_coor_new[0], max_coor_new[2]

    # max x, min z
    corner_4 = max_min_coor[0], max_min_coor[2]

    # min y
    height_min = min_coor_new[1]

    # max y
    height_max = max_coor_new[1]
    return [corner_1, corner_2, corner_3, corner_4, height_min, height_max]


def get_iou_cuboid(cu1, cu2):
    polygon_1 = Polygon([cu1[0], cu1[1], cu1[2], cu1[3]])
    polygon_2 = Polygon([cu2[0], cu2[1], cu2[2], cu2[3]])
    intersect_2d = polygon_1.intersection(polygon_2).area
    inter_vol = intersect_2d * max(0.0, min(cu1[5], cu2[5]) - max(cu1[4], cu2[4]))
    vol1 = polygon_1.area * (cu1[5] - cu1[4])
    vol2 = polygon_2.area * (cu2[5] - cu2[4])
    return inter_vol / (vol1 + vol2 - inter_vol + 0.00001)


def get_sg_from_words(objs_in_scene, rels_in_scene):
    relationships = [
        '__in_room__',
        'left of',
        'right of',
        'behind',
        'in front of',
        'inside',
        'surrounding',
        'left touching',
        'right touching',
        'front touching',
        'behind touching',
        'front left',
        'front right',
        'back left',
        'back right',
        'on']
    valid_classes = ["__room__", "curtain", "shower_curtain", "dresser", "counter", "bookshelf", "picture", "mirror",
                     "floor_mat", "chair", "sink", "desk", "table", "lamp", "door", "clothes", "person", "toilet",
                     "cabinet", "floor", "window", "blinds", "wall", "pillow", "whiteboard", "bathtub", "television",
                     "night_stand", "sofa", "refridgerator", "bed", "shelves"]
    obj_type_buffer = []
    rel_buffer = []
    # So you can do like chair:0, chair:1 etc etc
    for obj_name in objs_in_scene:
        if ":" in obj_name:
            new_obj_name = obj_name.split(":")[0]
        else:
            new_obj_name = obj_name
        obj_type_buffer.append(valid_classes.index(new_obj_name))

    for rel_idx in range(len(rels_in_scene)):
        obj_1 = rels_in_scene[rel_idx][0]
        obj_2 = rels_in_scene[rel_idx][2]
        rel_type = relationships.index(rels_in_scene[rel_idx][1])
        rel_buffer.append([objs_in_scene.index(obj_1), rel_type, objs_in_scene.index(obj_2)])

    # Add the dummy __in_room__ relationship
    for obj_idx in range(len(objs_in_scene)):
        rel_buffer.append([obj_idx, 0, len(objs_in_scene)])
    obj_type_buffer.append(0)
    # None attributes for all
    attributes = [0] * len(obj_type_buffer)
    attributes = torch.from_numpy(np.array(attributes)).long()
    triples = torch.from_numpy(np.array(rel_buffer)).long()
    objs = torch.from_numpy(np.array(obj_type_buffer)).long()
    return objs, triples, attributes


def random_scene(objs, boxes, angles):
    objs = objs.data.cpu()
    boxes = boxes.data.cpu()
    angles = angles.data.cpu()
    angles_rand = (torch.rand(angles.size()) * 24).type(torch.LongTensor)
    N = boxes.size(0)
    boxes_rand = []
    for i in range(N):
        if objs[i] != 0:
            box = boxes[i]
            x_c = np.random.rand()
            y_c = np.random.rand()
            z_c = np.random.rand()
            x0 = x_c - (box[3] - box[0]) / 2
            x1 = x_c + (box[3] - box[0]) / 2
            y0 = y_c - (box[4] - box[1]) / 2
            y1 = y_c + (box[4] - box[1]) / 2
            z0 = z_c - (box[5] - box[2]) / 2
            z1 = z_c + (box[5] - box[2]) / 2
            boxes_rand.append(torch.FloatTensor([x0, y0, z0, x1, y1, z1]))
        else:
            boxes_rand.append(boxes[i])
    boxes_rand = torch.stack(boxes_rand, dim=0)
    return boxes_rand, angles_rand


def restore_box(vocab, objs, boxes):
    b = boxes.size(0)
    prev = 0
    for cur in range(b):
        if vocab["object_idx_to_name"][objs[cur]] == "__room__":
            for i in range(prev, cur):
                boxes[i][0] *= boxes[cur][3]
                boxes[i][3] *= boxes[cur][3]
                boxes[i][1] *= boxes[cur][4]
                boxes[i][4] *= boxes[cur][4]
                boxes[i][2] *= boxes[cur][5]
                boxes[i][5] *= boxes[cur][5]
            prev = cur + 1
    return boxes


def scene_graph_acc(vocab, objs, triples, boxes):
    boxes = restore_box(vocab, objs, boxes)
    T = triples.size(0)
    good = 0
    for t in range(T):
        triple = triples[t]
        obj1 = triple[0]
        box1 = boxes[obj1]
        name1 = vocab["object_idx_to_name"][objs[obj1]]
        obj2 = triple[2]
        box2 = boxes[obj2]
        name2 = vocab["object_idx_to_name"][objs[obj2]]
        p_gt = vocab["pred_idx_to_name"][triple[1]]
        p = compute_rel(box1, box2, name1, name2)

        if p == p_gt:
            good += 1
    return good
