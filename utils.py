import time
import json

import numpy as np
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_model_losses(args, model, bbox, bbox_pred, angles, angles_pred, mu=None, logvar=None, KL_weight=None):
    dtype_f = bbox_pred.data.type()
    total_loss = 0.0
    losses = {}

    loss_bbox = F.l1_loss(bbox_pred, bbox)
    total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred', 1)

    loss_angle = F.nll_loss(angles_pred, angles)
    total_loss = add_loss(total_loss, loss_angle, losses, 'angle_pred', 1)

    if not args.use_AE:
        try:
            loss_gauss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        except:
            print("blowup!!!")
            print("logvar", torch.sum(logvar.data), torch.sum(torch.abs(logvar.data)), torch.max(logvar.data),
                  torch.min(logvar.data))
            print("mu", torch.sum(mu.data), torch.sum(torch.abs(mu.data)), torch.max(mu.data), torch.min(mu.data))
            return total_loss, losses
        total_loss = add_loss(total_loss, loss_gauss, losses, 'KLD_Gauss', KL_weight)
    return total_loss, losses


def compute_rel(box1, box2, name1, name2):
    center1 = np.array([(box1[0] + box1[3]) / 2, (box1[1] + box1[4]) / 2, (box1[2] + box1[5]) / 2])
    center2 = np.array([(box2[0] + box2[3]) / 2, (box2[1] + box2[4]) / 2, (box2[2] + box2[5]) / 2])

    if name2 == "__room__":
        p = "__in_room__"
    else:
        # "on" relationship
        p = None
        if center1[0] >= box2[0] and center1[0] <= box2[3]:
            if center1[2] >= box2[2] and center1[2] <= box2[5]:
                delta1 = center1[1] - center2[1]
                delta2 = (box1[4] - box1[1] + box2[4] - box2[1]) / 2
                if abs(delta1 - delta2) < 0.05:
                    p = 'on'
                    return p

        # random relationship
        sx0, sy0, sz0, sx1, sy1, sz1 = box1
        ox0, oy0, oz0, ox1, oy1, oz1 = box2
        d = center1 - center2
        theta = math.atan2(d[2], d[0])  # range -pi to pi

        area_s = (sx1 - sx0) * (sz1 - sz0)
        area_o = (ox1 - ox0) * (oz1 - oz0)
        ix0, ix1 = max(sx0, ox0), min(sx1, ox1)
        iz0, iz1 = max(sz0, oz0), min(sz1, oz1)
        area_i = max(0, ix1 - ix0) * max(0, iz1 - iz0)
        iou = area_i / (area_s + area_o - area_i)
        touching = 0.0001 < iou < 0.5

        if sx0 < ox0 and sx1 > ox1 and sz0 < oz0 and sz1 > oz1:
            p = 'surrounding'
        elif sx0 > ox0 and sx1 < ox1 and sz0 > oz0 and sz1 < oz1:
            p = 'inside'
        elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            p = 'right touching' if touching else 'left of'
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            p = 'behind touching' if touching else 'behind'
        elif -math.pi / 4 <= theta < math.pi / 4:
            p = 'left touching' if touching else 'right of'
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            p = 'front touching' if touching else 'in front of'

    return p


def load_json(json_file):
    with open(json_file, 'r') as f:
        var = json.load(f)
    return var


def write_json(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f)


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def tensor_aug(tensors, volatile=False, use_gpu=True):
    var_list = []
    for tensor in tensors:
        if use_gpu:
            var = tensor.cuda()
        else:
            var = tensor
        if volatile:
            var.requires_grad = False
        var_list.append(var)
    return tuple(var_list)


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss_weighted = curr_loss * weight
    loss_dict[loss_name] = curr_loss_weighted.item()
    if total_loss is not None:
        return total_loss + curr_loss_weighted
    else:
        return curr_loss_weighted
    return 0


def get_model_attr(_object, attr):
    if isinstance(_object, nn.DataParallel):
        return getattr(_object.module, attr)
    else:
        return getattr(_object, attr)
