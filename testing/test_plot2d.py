# Plot the top down view of an layout
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from testing.test_utils import get_eight_coors_bbox_new

def plot2d(boxes, angles, objs, save_path):
    valid_classes = ["__room__", "curtain", "shower_curtain", "dresser", "counter", "bookshelf", "picture", "mirror",
                     "floor_mat", "chair", "sink", "desk", "table", "lamp", "door", "clothes", "person", "toilet",
                     "cabinet", "floor", "window", "blinds", "wall", "pillow", "whiteboard", "bathtub", "television",
                     "night_stand", "sofa", "refridgerator", "bed", "shelves"]
    # Ignore the room element

    nyu_class_orig = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                      'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                      'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower_curtain',
                      'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag',
                      'otherstructure', 'otherfurniture', 'otherprop']

    # plot stuff that comes in later in the list on top
    # television generally is on top of stuff
    # Move bed to last, because it might overlap with other objs, easier for debugging
    nyu_class_order = ['wall', 'floor', 'cabinet', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                       'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                       'clothes', 'ceiling', 'books', 'refridgerator', 'paper', 'towel', 'shower_curtain',
                       'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag',
                       'otherstructure', 'otherfurniture', 'otherprop', 'television', 'bed']
    mapped_colors = [
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144)
    ]
    # Ignore structural elements (usually stuff on walls)
    # Also ignore the person class, because SUNCG doesn't have great human meshes
    do_not_vis = ["wall", "ceiling", "floor", "person", "door", "window", "curtain", "blinds", "__room__"]
    bbox_coors_min = []
    bbox_coors_min_max = []
    bbox_coors_max = []
    bbox_coors_max_min = []
    named_types = []
    # four corners, in order
    # remember, the middle dimension is top down
    # So we replace the 0th and 2nd dimension

    for obj_idx in range(len(objs)):
        model_type = valid_classes[int(objs[obj_idx])]
        if model_type in do_not_vis:
            continue
        bbox_min = boxes[obj_idx][:3] * boxes[-1][3:]
        bbox_max = boxes[obj_idx][3:] * boxes[-1][3:]
        obj_center = (bbox_max + bbox_min) / 2
        bbox_min = bbox_min - obj_center
        bbox_max = bbox_max - obj_center
        bbox_min_max = bbox_min.detach().clone()
        bbox_min_max[2] = bbox_max[2]
        bbox_max_min = bbox_min.detach().clone()
        bbox_max_min[0] = bbox_max[0]
        named_types.append(model_type)
        rot = torch.eye(3, dtype=torch.float)
        theta = -angles[obj_idx] * (2.0 * float(np.pi) / 24.0)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rot[0, 0] = cos_theta
        rot[0, 2] = sin_theta
        rot[2, 0] = -sin_theta
        rot[2, 2] = cos_theta
        trans = obj_center.numpy()
        bbox_coors_min.append(torch.matmul(rot, bbox_min).detach().cpu().numpy()+trans)
        bbox_coors_min_max.append(torch.matmul(rot, bbox_min_max).detach().cpu().numpy()+trans)
        bbox_coors_max.append(torch.matmul(rot, bbox_max).detach().cpu().numpy()+trans)
        bbox_coors_max_min.append(torch.matmul(rot, bbox_max_min).detach().cpu().numpy()+trans)
    fig, ax = plt.subplots()
    patches = []
    colors = []
    # Make the floor first
    polygon = Polygon(np.array([[-0.1, -0.1], [-0.1, 1.1], [1.1, 1.1], [1.1, -0.1]]), True)
    patches.append(polygon)
    colors.append(mapped_colors[nyu_class_orig.index("floor")])
    current_types = list(map(lambda x: nyu_class_order.index(x), named_types))
    iter_idx = list(range(len(current_types)))
    iter_idx = [x for _, x in sorted(zip(current_types, iter_idx))]
    for box_id in iter_idx:
        colors.append(mapped_colors[nyu_class_orig.index(named_types[box_id])])
        cur_box = np.array(get_eight_coors_bbox_new(bbox_coors_min[box_id], bbox_coors_max[box_id], bbox_coors_min_max[box_id], bbox_coors_max_min[box_id])[:4])
        cur_box[:, 1] = 1 - cur_box[:, 1]
        polygon = Polygon(np.array(cur_box), True)
        patches.append(polygon)
    colors = np.array(colors) / 255.0
    alpha = np.ones((len(colors), 1))
    colors = np.hstack((np.array(colors), alpha))
    p = PatchCollection(patches, facecolors=colors, alpha=1.0)
    ax.add_collection(p)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.set(aspect='equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.tight_layout()
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.savefig(save_path)
    plt.close()





