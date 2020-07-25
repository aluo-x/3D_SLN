import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
from utils import load_json, compute_rel

class SuncgDataset(BaseDataset):
    def __init__(self, data_dir, train_3d, touching_relations=True, use_attr_30=False):
        super(Dataset, self).__init__()
        self.train_3d = train_3d
        assert self.train_3d
        # Do we train using 3D coors? You want True.

        self.use_attr_30 = use_attr_30
        # Do we want to train on object attributes? Split by 70:30? Tall/Short & Large/Small & None?
        print("Starting to read the json file for SUNCG")
        self.data = load_json(data_dir)
        # Json file for cleaned & normalized data

        self.room_ids = [int(i) for i in list(self.data)]

        self.touching_relations = touching_relations
        # Do objects touch? Works either way

        # Construction dict
        # obj_name is object type (chair/table/sofa etc. etc.)
        # pred_name is relation type (left/right etc.)
        # idx_to_name maps respective index back to object type or relation name
        valid_types = load_json("metadata/valid_types.json")
        self.vocab = {'object_idx_to_name': ['__room__'] + valid_types}

        # map obj type to idx
        self.vocab['object_name_to_idx'] = {}
        for i, name in enumerate(self.vocab['object_idx_to_name']):
            self.vocab['object_name_to_idx'][name] = i

        # map idx to relation type
        self.vocab['pred_idx_to_name'] = [
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
            'on',
        ]
        # We don't actually use the front left, front right, back left, back right

        # map relation type to idx
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

        self.vocab['attrib_idx_to_name'] = [
            'none',
            'tall',
            'short',
            'large',
            'small',
        ]
        self.vocab['attrib_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['attrib_idx_to_name']):
            self.vocab['attrib_name_to_idx'][name] = idx


        self.image_id_to_objects = defaultdict(list)
        self.room_bboxes = {}
        for room_id in self.data:
            room = self.data[room_id]
            room_id = int(room_id)
            self.image_id_to_objects[room_id] = room["valid_objects"]
            self.room_bboxes[room_id] = room["bbox"]

        self.size_data = load_json(
            "metadata/size_info_many.json")
        self.size_data_30 = load_json(
            "metadata/30_size_info_many.json")

    def total_objects(self):
        total_objs = 0
        for i, room_id in enumerate(self.room_ids):
            num_objs = len(self.image_id_to_objects[room_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        return len(self.room_ids)

    def return_room_ids(self):
        return self.room_ids

    def get_by_room_id(self, room_id):
        try:
            idx = self.room_ids.index(int(room_id))
        except:
            print("Get by room id failed! Defaulting to 0.")
            idx = 0
        return self.__getitem__(idx)

    def __getitem__(self, index):
        room_id = self.room_ids[index]
        objs, boxes, angles = [], [], []
        for object_data in self.image_id_to_objects[room_id]:
            obj_type = object_data["type"]
            objs.append(self.vocab['object_name_to_idx'][obj_type])
            bbox = object_data['new_bbox']
            # Get min/max of the bbox
            x0 = bbox[0][0]
            y0 = bbox[0][1]
            z0 = bbox[0][2]
            x1 = bbox[1][0]
            y1 = bbox[1][1]
            z1 = bbox[1][2]
            if self.train_3d:
                boxes.append(torch.FloatTensor([x0, y0, z0, x1, y1, z1]))
            else:
                boxes.append(torch.FloatTensor([x0, z0, x1, z1]))

            theta = object_data['rotation']
            angles.append(theta)

        objs.append(self.vocab['object_name_to_idx']['__room__'])
        room_bbox = self.room_bboxes[room_id]
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        x1 = room_bbox[0]
        y1 = room_bbox[1]
        z1 = room_bbox[2]
        if self.train_3d:
            boxes.append(torch.FloatTensor([x0, y0, z0, x1, y1, z1]))
        else:
            boxes.append(torch.FloatTensor([x0, z0, x1, z1]))
        angles.append(0)

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        # Angles are discrete, so make it a long tensor
        angles = torch.LongTensor(angles)

        # Compute centers of all objects
        obj_centers = []
        if self.train_3d:
            for i, obj_idx in enumerate(objs):
                x0, y0, z0, x1, y1, z1 = boxes[i]
                mean_x = 0.5 * (x0 + x1)
                mean_y = 0.5 * (y0 + y1)
                mean_z = 0.5 * (z0 + z1)
                obj_centers.append([mean_x, mean_y, mean_z])
        else:
            for i, obj_idx in enumerate(objs):
                x0, z0, x1, z1 = boxes[i]
                mean_x = 0.5 * (x0 + x1)
                mean_z = 0.5 * (z0 + z1)
                obj_centers.append([mean_x, mean_z])
        obj_centers = torch.FloatTensor(obj_centers)

        # Compute scene graphs
        triples = []
        num_objs = objs.size(0)
        __room__ = self.vocab['object_name_to_idx']['__room__']
        real_objs = []

        if num_objs > 1:
            # get non-room object indices
            real_objs = (objs != __room__).nonzero().squeeze(1)

        if self.train_3d:
            # special: "on" relationships
            on_rels = defaultdict(list)
            for cur in real_objs:
                choices = [obj for obj in real_objs if obj != cur]
                for other in choices:
                    cur_box = boxes[cur]
                    other_box = boxes[other]
                    p = compute_rel(cur_box, other_box, None, None)
                    if p == "on":
                        p = self.vocab['pred_name_to_idx']['on']
                        triples.append([cur, p, other])
                        on_rels[cur].append(other)

            # add random relationships
            for cur in real_objs:
                choices = [obj for obj in real_objs if obj != cur]
                other = random.choice(choices)
                if random.random() > 0.5:
                    s, o = cur, other
                else:
                    s, o = other, cur
                if s in on_rels[o] or o in on_rels[s]:
                    continue

                p = compute_rel(boxes[s], boxes[o], None, None)
                p = self.vocab['pred_name_to_idx'][p]
                triples.append([s, p, o])

            # Add __in_room__ triples
            O = objs.size(0)
            for i in range(O - 1):
                p = compute_rel(boxes[i], boxes[-1], None, "__room__")
                p = self.vocab['pred_name_to_idx'][p]
                triples.append([i, p, O - 1])

        triples = torch.LongTensor(triples)

        # normalize boxes, all in [0,1] relative to room
        b = boxes.size(0)
        if self.train_3d:
            for i in range(b - 1):
                boxes[i][0] /= boxes[-1][3]
                boxes[i][3] /= boxes[-1][3]
                boxes[i][1] /= boxes[-1][4]
                boxes[i][4] /= boxes[-1][4]
                boxes[i][2] /= boxes[-1][5]
                boxes[i][5] /= boxes[-1][5]
        else:
            for i in range(b - 1):
                boxes[i][0] /= boxes[-1][2]
                boxes[i][2] /= boxes[-1][2]
                boxes[i][1] /= boxes[-1][3]
                boxes[i][3] /= boxes[-1][3]

        if not self.use_attr_30:
            # compute size attributes using normalized bboxes
            attributes = []
            for i in range(b - 1):
                obj_type = self.vocab['object_idx_to_name'][objs[i]]
                if random.random() > 0.5 or (obj_type not in self.size_data):
                    attributes.append("none")
                else:
                    obj_type = self.vocab['object_idx_to_name'][objs[i]]
                    if random.random() > 0.5:
                        # tall/short
                        obj_height = boxes[i][4] - boxes[i][1]
                        if obj_height > self.size_data[obj_type][0][1]:
                            attributes.append("tall")
                        else:
                            attributes.append("short")
                    else:
                        # large/small
                        obj_volume = (boxes[i][3] - boxes[i][0]) * (boxes[i][4] - boxes[i][1]) * (
                                    boxes[i][5] - boxes[i][2])
                        if obj_volume > self.size_data[obj_type][1]:
                            attributes.append("large")
                        else:
                            attributes.append("small")
        else:
            # compute size attributes using normalized bboxes, use 30/70 size
            attributes = []
            for i in range(b - 1):
                obj_type = self.vocab['object_idx_to_name'][objs[i]]
                if random.random() > 0.5 or (obj_type not in self.size_data_30):
                    # if random.random() > 0.7:
                    attributes.append("none")
                else:
                    obj_type = self.vocab['object_idx_to_name'][objs[i]]
                    if random.random() > 0.5:
                        # tall/short
                        obj_height = boxes[i][4] - boxes[i][1]
                        if obj_height > self.size_data_30[obj_type]["height_7"]:
                            attributes.append("tall")
                        elif obj_height < self.size_data_30[obj_type]["height_3"]:
                            attributes.append("short")
                        else:
                            attributes.append("none")
                    else:
                        # large/small
                        obj_volume = (boxes[i][3] - boxes[i][0]) * (boxes[i][4] - boxes[i][1]) * (
                                    boxes[i][5] - boxes[i][2])
                        if obj_volume > self.size_data_30[obj_type]["volume_7"]:
                            attributes.append("large")
                        elif obj_volume < self.size_data_30[obj_type]["volume_3"]:
                            attributes.append("small")
                        else:
                            attributes.append("none")

        attributes.append("none")
        attributes = [self.vocab["attrib_name_to_idx"][name] for name in attributes]
        attributes = torch.LongTensor(attributes)

        assert attributes.size(0) == objs.size(0)
        return room_id, objs, boxes, triples, angles, attributes


def suncg_collate_fn(batch):
    """
    Collate function to be used when wrapping SuncgDataset in a
    DataLoader. Returns a tuple of the following:

    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to room
    - triple_to_img: LongTensor of shape (T,) mapping triples to room
    """
    all_ids, all_objs, all_boxes, all_triples, all_angles, all_attributes = [], [], [], [], [], []
    all_obj_to_room, all_triple_to_room = [], []
    obj_offset = 0
    for i, (room_id, objs, boxes, triples, angles, attributes) in enumerate(batch):
        if objs.dim() == 0 or triples.dim() == 0:
            continue
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_angles.append(angles)
        all_attributes.append(attributes)
        all_boxes.append(boxes)
        all_ids.append(room_id)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_room.append(torch.LongTensor(O).fill_(i))
        all_triple_to_room.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_ids = torch.LongTensor(all_ids)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_angles = torch.cat(all_angles)
    all_attributes = torch.cat(all_attributes)
    all_obj_to_room = torch.cat(all_obj_to_room)
    all_triple_to_room = torch.cat(all_triple_to_room)

    out = (all_ids, all_objs, all_boxes, all_triples, all_angles, all_attributes, all_obj_to_room, all_triple_to_room)
    return out
