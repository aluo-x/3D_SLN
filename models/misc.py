from options.options import Options
args = Options().parse()

import json
import numpy as np
import os
import neural_renderer as nr
# Used in diff_render.py

# TODO: remove this once I fix the environment
import torch
# This is the code used to help the neural renderer
def load_json(json_file):
    with open(json_file,'r') as f:
        var = json.load(f)
    return var
import pywavefront as pwf
# For loading meshes

import pymesh
# For re-meshing objects

suncg_obj_dir = os.path.join(args.suncg_data_dir, "object")
suncg_room_dir = os.path.join(args.suncg_data_dir, "room")

suncg_valid_types = load_json("metadata/valid_types.json")

object_idx_to_name = ['__room__'] + suncg_valid_types
wall_data_json = load_json("metadata/wall_data_wfc.json")

suncg_data = load_json("metadata/suncg_data_many.json")


def suncg_retrieve(objs, bboxes):
    input_boxes = [box.cpu().detach().numpy() for box in bboxes]
    for i_box in range(len(input_boxes[:-1])):
        input_boxes[i_box][0] *= input_boxes[-1][3]
        input_boxes[i_box][3] *= input_boxes[-1][3]
        input_boxes[i_box][1] *= input_boxes[-1][4]
        input_boxes[i_box][4] *= input_boxes[-1][4]
        input_boxes[i_box][2] *= input_boxes[-1][5]
        input_boxes[i_box][5] *= input_boxes[-1][5]
    O = len(objs)
    # object_idx_to_name = ['__room__'] + valid_types
    ids = []
    for i in range(O - 1):
        obj_idx = objs[i]
        obj_type = object_idx_to_name[obj_idx]
        box = input_boxes[i]
        deltax = box[3] - box[0]
        deltay = box[4] - box[1]
        deltaz = box[5] - box[2]
        ratio = np.array([deltay / deltax, deltaz / deltax])

        obj_data = suncg_data[obj_type]
        ratio_data = []
        for obj in obj_data:
            obj_size = np.array(obj["bbox_max"]) - np.array(obj["bbox_min"])
            obj_ratio = np.array([obj_size[1] / obj_size[0], obj_size[2] / obj_size[0]])
            ratio_data.append(np.sum(np.abs(obj_ratio - ratio)))

        obj_choose = np.argmin(ratio_data)
        ids.append(obj_data[obj_choose]["id"])
    return ids

def custom_load_obj(filename_obj):
    try:
        obj_info = pwf.Wavefront(filename_obj, strict=1, collect_faces=True)
    except Exception as e:
        print("Loading obj failed inside new load func")
        print(e)
        return np.array([]).astype(np.float32), np.array([]).astype(np.int32)
    vert = obj_info.vertices
    total_mesh = []
    mesh_buffer = obj_info.mesh_list
    num_mesh = len(mesh_buffer)
    for mesh_id in range(num_mesh):
        total_mesh = total_mesh + mesh_buffer[mesh_id].faces
    output_vertices, output_faces, info = pymesh.split_long_edges_raw(np.array(vert).astype(np.float32), np.array(total_mesh).astype(np.int32), 0.6)
    return output_vertices.astype(np.float32), output_faces.astype(np.int32)

def custom_load_wall(filename_obj):
    try:
        obj_info = pwf.Wavefront(filename_obj, strict=1, collect_faces=True)
    except Exception as e:
        print("Loading obj failed inside new load func")
        print(e)
        return np.array([]).astype(np.float32), np.array([]).astype(np.int32)
    vert = obj_info.vertices
    total_mesh = []
    total_vert = []
    mesh_buffer = obj_info.mesh_list
    num_mesh = len(mesh_buffer)
    for mesh_id in range(num_mesh):
        total_mesh.append(mesh_buffer[mesh_id].faces)
        total_vert.append(vert)
    remesh_vert = []
    remesh_face = []
    for mesh_id in range(num_mesh):
        output_vertices, output_faces, info = pymesh.split_long_edges_raw(np.array(total_vert[mesh_id]).astype(np.float32), np.array(total_mesh[mesh_id]).astype(np.int32), 0.6)
        # Remesh operation maintains offsets
        # So if orig sub mesh face indices was 32 ~ 77, then it will still start at 32, but end at some larger number
        # We work around this by remeshing each set of faces (and all the vertices)
        # Then basically treat them as different meshes
        remesh_vert.append(output_vertices.astype(np.float32))
        remesh_face.append(output_faces.astype(np.int32))
    return remesh_vert, remesh_face

cache_dict = {}

def load_suncg_obj(model_id):
    # print("Loading {}".format(str(model_id)))
    global cache_dict
    model_path = os.path.join(suncg_obj_dir, model_id, model_id + ".obj")
    id_name = model_path
    if id_name in cache_dict:
        vertices, faces = cache_dict[id_name]
        return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()
    vertices, faces = custom_load_obj(model_path)
    cache_dict[id_name] = [vertices, faces]
    return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()

def wall_retrieve(boxes, data=wall_data_json):
    np_boxes = [box.cpu().detach().numpy().astype("float") for box in boxes]
    X = np_boxes[-1][3]
    Y = np_boxes[-1][4]
    Z = np_boxes[-1][5]
    ratio = np.array([Y / X, Z / X], dtype=np.float)

    ratio_data = []
    for wall in data:
        wall_size = np.array(wall["wall_bbox_max"], dtype=np.float) - np.array(wall["wall_bbox_min"], dtype=np.float)
        wall_ratio = np.array([wall_size[1] / wall_size[0], wall_size[2] / wall_size[0]], dtype=np.float)
        ratio_data.append(np.sum(np.abs(wall_ratio - ratio)))
    wall_choose = np.argmin(ratio_data)
    wall = data[wall_choose]
    return wall

def floor_retrieve(boxes, data=wall_data_json):
    # wall data json is correct here
    np_boxes = [box.cpu().detach().numpy().astype("float") for box in boxes]
    X = np_boxes[-1][3]
    Z = np_boxes[-1][5]
    ratio = Z / X
    ratio_data = []
    for floor in data:
        floor_size = np.array(floor["floor_bbox_max"]) - np.array(floor["floor_bbox_min"], dtype=np.float)
        floor_ratio = floor_size[2] / floor_size[0]
        ratio_data.append(np.abs(floor_ratio - ratio))
    floor_choose = np.argmin(ratio_data)
    floor = data[floor_choose]
    return floor

def load_wall_obj_new(wall_data):
    model_path = os.path.join(suncg_room_dir, wall_data["house_id"], wall_data["model_id"] + "w.obj")
    vertices, faces = custom_load_wall(model_path)
    new_vertices = []
    new_faces = []
    for mesh_id in range(len(vertices)):
        new_vertices.append(torch.from_numpy(vertices[mesh_id]).cuda())
        new_faces.append(torch.from_numpy(faces[mesh_id]).cuda())
    return new_vertices, new_faces

def load_floor_obj(floor_data):
    model_path = os.path.join(suncg_room_dir, floor_data["house_id"], floor_data["model_id"] + "f.obj")
    # print("Loading floor: {}".format(str(model_path)))
    global cache_dict
    id_name = model_path
    if id_name in cache_dict:
        print("loading from cache")
        vertices, faces = cache_dict[id_name]
        return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()
    vertices, faces = custom_load_obj(model_path)
    cache_dict[id_name] = [vertices, faces]
    return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()

def load_ceil_obj(wall_data):
    model_path = os.path.join(suncg_room_dir, wall_data["house_id"], wall_data["model_id"] + "c.obj")
    if not os.path.exists(model_path):
        print("Missing ceiling")
        raise ValueError
    # print("Loading ceiling: {}".format(str(model_path)))
    global cache_dict
    id_name = model_path
    if id_name in cache_dict:
        print("loading from cache")
        vertices, faces = cache_dict[id_name]
        return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()
    vertices, faces = custom_load_obj(model_path)
    cache_dict[id_name] = [vertices, faces]
    return torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda()

def get_bbox(vertex_data):
    return torch.max(vertex_data, dim=0)[0], torch.min(vertex_data, dim=0)[0]