import numpy as np
import random
import sys
import copy
import os
sys.path.append(os.getcwd())

from xiuminglib.blender import camera, lighting
from xiuminglib.blender.object import *
from xiuminglib.blender.scene import set_cycles
import bpy
import subprocess
import json
def load_json(json_file):
    with open(json_file, 'r') as f:
        var = json.load(f)
    return var

import os
suncg_obj_dir = os.environ["SUNCG_DIR"]+"/object"
suncg_room_dir = os.environ["SUNCG_DIR"]+"/room"
vocab = load_json("./metadata/vocab_many.json")

object_idx_to_name = vocab["object_idx_to_name"]
suncg_data = load_json("./metadata/suncg_data_many.json")
wall_data = load_json("./metadata/wall_data_wfc.json")


def suncg_retrieve(data, objs, bboxes):
    O = len(objs)
    # object_idx_to_name = ['__image__'] + valid_types
    ids = []
    for i in range(O - 1):
        obj_idx = objs[i]
        obj_type = object_idx_to_name[obj_idx]
        box = bboxes[i]
        deltax = box[3] - box[0]
        deltay = box[4] - box[1]
        deltaz = box[5] - box[2]
        ratio = np.array([deltay / deltax, deltaz / deltax])

        obj_data = data[obj_type]
        ratio_data = []
        for obj in obj_data:
            obj_size = np.array(obj["bbox_max"]) - np.array(obj["bbox_min"])
            obj_ratio = np.array([obj_size[1] / obj_size[0], obj_size[2] / obj_size[0]])
            ratio_data.append(np.sum(np.abs(obj_ratio - ratio)))

        obj_choose = np.argmin(ratio_data)
        ids.append(obj_data[obj_choose]["id"])
    return ids



def wall_retrieve(boxes, data=wall_data):
    X = boxes[-1][3]
    Y = boxes[-1][4]
    Z = boxes[-1][5]
    ratio = np.array([Y / X, Z / X])

    ratio_data = []
    for wall in data:
        wall_size = np.array(wall["wall_bbox_max"]) - np.array(wall["wall_bbox_min"])
        wall_ratio = np.array([wall_size[1] / wall_size[0], wall_size[2] / wall_size[0]])
        ratio_data.append(np.sum(np.abs(wall_ratio - ratio)))
    wall_choose = np.argmin(ratio_data)
    wall = data[wall_choose]
    return wall


def floor_retrieve(boxes, data=wall_data):
    X = boxes[-1][3]
    Y = boxes[-1][4]
    Z = boxes[-1][5]
    ratio = Z / X

    ratio_data = []
    for floor in data:
        floor_size = np.array(floor["floor_bbox_max"]) - np.array(floor["floor_bbox_min"])
        floor_ratio = floor_size[2] / floor_size[0]
        ratio_data.append(np.abs(floor_ratio - ratio))
    floor_choose = np.argmin(ratio_data)
    floor = data[floor_choose]
    return floor


def assign_texture(obj, source="GAN", sample=True, img_dir=None):
    for mat in obj.data.materials:
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes

        # Remove all nodes
        for node in nodes:
            nodes.remove(node)

        texture = mat.active_texture

        if texture is not None:
            # get texture image name
            img = texture.image

            # Bundled texture found -- set up diffuse texture node tree
            nodes.new('ShaderNodeTexImage')
            print("adding texture image", img)
            nodes['Image Texture'].image = img
            nodes.new('ShaderNodeBsdfDiffuse')
            nodes.new('ShaderNodeOutputMaterial')
            node_tree.links.new(nodes['Image Texture'].outputs[0], nodes['Diffuse BSDF'].inputs[0])
            node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

        # if color is not None:
        #     logging.warning("%s: %s has a texture map associated with it -- 'color' argument ignored",
        #                     thisfunc, obj.name)

        else:
            # No texture found -- set up diffuse color tree
            nodes.new('ShaderNodeBsdfDiffuse')
            color = mat.diffuse_color
            color_rgba = (color[0], color[1], color[2], 1)
            nodes['Diffuse BSDF'].inputs[0].default_value = color_rgba
            nodes.new('ShaderNodeOutputMaterial')
            node_tree.links.new(nodes['Diffuse BSDF'].outputs[0], nodes['Material Output'].inputs[0])

    # Roughness
    # node_tree.nodes['Diffuse BSDF'].inputs[1].default_value = roughness

    # Scene update necessary, as matrix_world is updated lazily
    scene = bpy.context.scene
    scene.update()


def assign_texture_scene(room_objs, walls, floor, ceiling, option="original"):
    if option == "original":
        for obj in room_objs:
            assign_texture(obj, sample=False)
        for obj in walls:
            assign_texture(obj, sample=False)
        assign_texture(floor, sample=False)
        assign_texture(ceiling, sample=False)
    if option == "GAN":
        print("Not implemented")
        raise ValueError
    if option == "OpenSurfaces":
        print("Not implemented")
        raise ValueError


def render_test(objs, boxes, angles, out_path, name="", triples=None, dataset="suncg"):
    # restore boxes to real size
    boxes = copy.deepcopy(boxes)
    b = len(boxes)
    boxes[-1][3] -= boxes[-1][0]
    boxes[-1][4] -= boxes[-1][1]
    boxes[-1][5] -= boxes[-1][2]
    for i in range(b - 1):
        boxes[i][0] *= boxes[-1][3]
        boxes[i][3] *= boxes[-1][3]
        boxes[i][1] *= boxes[-1][4]
        boxes[i][4] *= boxes[-1][4]
        boxes[i][2] *= boxes[-1][5]
        boxes[i][5] *= boxes[-1][5]

        # height adjustment
        if abs(boxes[i][1]) <= 0.02:
            boxes[i][4] -= boxes[i][1]
            boxes[i][1] = 0

    room_bbox = boxes[-1][3:]

    # object retrieve
    if dataset == "suncg":
        ids = suncg_retrieve(suncg_data, objs, boxes)
    else:
        raise ValueError

    # wall,floor,ceiling retrieve
    wall_data = wall_retrieve(boxes)

    # scene initialization
    # reset_blend()
    bpy.ops.wm.read_factory_settings()
    scene = bpy.context.scene
    scene.render.filepath = out_path
    mainfile_path = "./mainfile"
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()
    scene.update()
    world = bpy.data.worlds.new(name="World")
    set_cycles(n_samples=50)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 25

    # add objects
    rotations = []
    translations = []
    centers = []
    sizes = []
    for i in range(b - 1):
        # get object size and rotation
        bbox_min = np.array(boxes[i][:3])
        bbox_max = np.array(boxes[i][3:])
        obj_center = (bbox_max + bbox_min) / 2
        obj_size = bbox_max - bbox_min
        theta = angles[i] * (2 * np.pi / 24)
        if dataset == "shapenet":
            theta += np.pi

        # get model data
        model_type = object_idx_to_name[objs[i]]
        if dataset == "suncg":
            for obj_model in suncg_data[model_type]:
                if obj_model["id"] == ids[i]:
                    model_bbox_min = np.array(obj_model["bbox_min"])
                    model_bbox_max = np.array(obj_model["bbox_max"])
                    model_size = model_bbox_max - model_bbox_min
                    model_center = (model_bbox_min + model_bbox_max) / 2
            scale = min([obj_size[0] / model_size[0], obj_size[1] / model_size[1], obj_size[2] / model_size[2]])

        # get object transform
        scaled_size = scale * model_size
        rot = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        obj_center[1] -= (obj_size[1] - scaled_size[1]) / 2
        trans = obj_center - scale * np.matmul(rot, model_center)
        rot *= scale
        centers.append(obj_center)
        sizes.append(scaled_size)
        rotations.append(rot)
        translations.append(trans)

    # import objects
    room_objs = []
    for i in range(b - 1):
        model_type = object_idx_to_name[objs[i]]
        if model_type in ["wall", "ceiling", "floor", "person", "door", "window", "curtain", "blinds"]:
            # Ignore non-furniture objects
            continue
        try:
            model_id = ids[i]
            rot = rotations[i]
            trans = translations[i]
            # import object
            if dataset == "suncg":
                model_path = os.path.join(suncg_obj_dir, model_id, model_id + ".obj")
            obj = import_object(model_path, rot_mat=rot, trans_vec=trans, scale=1, name=model_type)
            room_objs.append(obj)

            # update the scene
            bpy.context.scene.update()
        except:
            print("Load obj mesh failed")
            continue

    walls = []

    wall_bbox = np.array(boxes[-1][3:])
    obj_center = wall_bbox / 2
    obj_size = wall_bbox
    model_bbox_min = np.array(wall_data["wall_bbox_min"])
    model_bbox_max = np.array(wall_data["wall_bbox_max"])
    model_size = model_bbox_max - model_bbox_min
    model_center = (model_bbox_min + model_bbox_max) / 2
    scale = max([obj_size[0] / model_size[0], obj_size[1] / model_size[1], obj_size[2] / model_size[2]])
    rot = np.eye(3)
    trans = obj_center - scale * np.matmul(rot, model_center)
    rot *= scale
    model_path = os.path.join(suncg_room_dir, wall_data["house_id"], wall_data["model_id"] + "w.obj")
    objs = import_object(model_path, rot_mat=rot, trans_vec=trans, scale=1, name="wall")
    for obj in objs:
        # get rid of front wall, as well as walls inside room
        scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        verts = []
        score = 0
        total = 0
        for v in bm.verts:
            total += 1
            v_scene = obj.matrix_world * v.co
            # print(v_scene[2]/room_bbox[2],v_scene[0]/room_bbox[0])

            # if v_scene[2] > 0.9 * room_bbox[2]:
            if v_scene[2] > 0.2 * room_bbox[2]:
                if v_scene[2] > 0.9 * room_bbox[2]:
                    score += 1
                if v_scene[0] > 0.1 * room_bbox[0] and v_scene[0] < 0.9 * room_bbox[0]:
                    # print(v_scene)
                    verts.append(v)
        if score / len(bm.verts) > 0.7:
            verts = bm.verts
        # print(len(verts))
        bmesh.ops.delete(bm, geom=verts, context=1)
        bmesh.update_edit_mesh(me)
        bpy.ops.object.mode_set(mode='OBJECT')

    walls = objs

    bpy.context.scene.update()

    # add floor
    floor_bbox = np.array(boxes[-1][3:])
    obj_center = floor_bbox / 2
    obj_size = floor_bbox
    model_bbox_min = np.array(wall_data["floor_bbox_min"])
    model_bbox_max = np.array(wall_data["floor_bbox_max"])
    model_size = model_bbox_max - model_bbox_min
    model_center = (model_bbox_min + model_bbox_max) / 2
    scale = max([obj_size[0] / model_size[0], obj_size[2] / model_size[2]])
    scaled_size = scale * model_size
    rot = np.eye(3)
    obj_center[1] = -0.5 * scaled_size[1]
    trans = obj_center - scale * np.matmul(rot, model_center)
    rot *= scale
    model_path = os.path.join(suncg_room_dir, wall_data["house_id"], wall_data["model_id"] + "f.obj")
    obj = import_object(model_path, rot_mat=rot, trans_vec=trans, scale=1, name="floor")

    floor = obj

    bpy.context.scene.update()

    # add ceiling
    ceiling_bbox = np.array(boxes[-1][3:])
    obj_center = ceiling_bbox / 2
    obj_size = ceiling_bbox
    model_bbox_min = np.array(wall_data["ceiling_bbox_min"])
    model_bbox_max = np.array(wall_data["ceiling_bbox_max"])
    model_size = model_bbox_max - model_bbox_min
    model_center = (model_bbox_min + model_bbox_max) / 2
    scale = max([obj_size[0] / model_size[0], obj_size[2] / model_size[2]])
    scaled_size = scale * model_size
    rot = np.eye(3)
    obj_center[1] = 0.5 * scaled_size[1] + ceiling_bbox[1]
    trans = obj_center - scale * np.matmul(rot, model_center)
    rot *= scale
    model_path = os.path.join(suncg_room_dir, wall_data["house_id"], wall_data["model_id"] + "c.obj")
    obj = import_object(model_path, rot_mat=rot, trans_vec=trans, scale=1, name="ceiling")

    # assign_texture(obj, sample=False)

    ceiling = obj

    bpy.context.scene.update()

    # start viewpoint sampling
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 25

    Nsample = 5
    succeed = False

    for k in range(Nsample):
        cam_data = {}
        t = 0.2 + 0.6 * np.random.rand()
        cam_coord = np.array([t * room_bbox[0], 0.9 * room_bbox[1], room_bbox[2] + 0.4])
        f_mm = 50
        canonical_angle = np.pi / 2 - np.arctan(0.4 / (0.9 * room_bbox[1])) - np.arctan(25 / f_mm)
        canonical_angle -= np.random.rand() * 0.1
        plane_angle = np.arctan((cam_coord[0] - 0.5 * room_bbox[0]) / cam_coord[2]) * 1.1
        cam = camera.add_camera(xyz=cam_coord, rot_vec_rad=(-canonical_angle, plane_angle, 0), f=f_mm, sensor_width=50,
                                sensor_height=50, sensor_fit='VERTICAL')
        scene.camera = cam

        zbuffer = camera.get_camera_zbuffer(cam)
        dist = 0
        ndist = 0
        for i in range(zbuffer.shape[0]):
            for j in range(zbuffer.shape[1]):
                if zbuffer[i][j] < 1e5:
                    dist += zbuffer[i][j]
                    ndist += 1
        mean_depth = dist / ndist
        # print("mean depth:",mean_depth,room_id)
        if mean_depth > 0.7:
            succeed = True
            break

    if not succeed:
        print("Failed to sample good view point")
        return None

    sideview = "front"
    if sideview == "side":
        cam_coord = np.array([-0.4, 1, 2])
        f_mm = 80
        cam = camera.add_camera(xyz=cam_coord, rot_vec_rad=(-np.pi / 8, -np.pi / 6, 0), f=f_mm, sensor_width=50,
                                sensor_height=int(50 * boxes[-1][5]), sensor_fit='VERTICAL')
        scene.render.resolution_x = 1500
        scene.render.resolution_y = 1000
        scene.camera = cam
    if sideview == "left":
        cam_coord = np.array([-0.8, boxes[-1][4] / 2, boxes[-1][5] / 2])
        f_mm = 80
        cam = camera.add_camera(xyz=cam_coord, rot_vec_rad=(0, -np.pi / 2, 0), f=f_mm, sensor_width=50,
                                sensor_height=int(50 * boxes[-1][5]), sensor_fit='VERTICAL')
        scene.render.resolution_x = int(1000 * boxes[-1][5])
        scene.render.resolution_y = int(1000 * boxes[-1][4])
        scene.camera = cam

    # add area light

    cur_light = lighting.add_light_area(xyz=(room_bbox[0] / 2, room_bbox[1] * 0.9, room_bbox[2] / 2),
                                        rot_vec_rad=(0, 0, 0), name=None, energy=1.2, size=0.1)

    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    nodes = tree.nodes
    links = tree.links

    node_bg = nodes["Background"]
    node_bg.inputs["Strength"].default_value = 1.0

    node_env = nodes.new(type='ShaderNodeTexEnvironment')
    hdr_dir = "./metadata/hdr_image/"
    hdr_image = random.choice(os.listdir(hdr_dir))
    node_env.image = bpy.data.images.load(os.path.join(hdr_dir, hdr_image))
    node_env.projection = 'EQUIRECTANGULAR'  # or EQUIRECTANGULAR
    node_env.texture_mapping.scale = (0.1, 0.1, 0.1)
    node_env.texture_mapping.rotation = (np.pi / 2, 0, np.random.rand() * 2 * np.pi)
    # node_env.texture_mapping.mapping_z = 'Y'
    links.new(node_env.outputs["Color"], node_bg.inputs["Color"])
    node_bg.location = (0.5, 1000)
    # node_bg.texture_mapping.scale = (0.1,0.1,0.1)
    cycles = world.cycles
    cycles.sample_as_light = True
    cycles.sample_map_resolution = 512
    # render scene
    assign_texture_scene(room_objs, walls, floor, ceiling, option="original")
    scene.render.filepath = os.path.join(out_path, name)
    bpy.ops.render.render(write_still=True)


    # clear memory by reloading mainfile
    bpy.ops.wm.save_as_mainfile(filepath=mainfile_path)
    bpy.ops.wm.open_mainfile(filepath=mainfile_path)
    subprocess.run(["rm", mainfile_path])

    # log.close()
    return 0