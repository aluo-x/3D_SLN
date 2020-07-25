from models.misc import *
from torch import nn
nyu_class = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator','television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
inter_out = 512
final_out = 256

import pickle
def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return True

def get_cam_mat(boxes):
    # TODO: old parameters were theta_rot = -0.3, offset is 0.18 instead of 0.16
    theta_rot = -0.4
    # this is kinda a hack, tweak the 400 (to 600?) as needed to change the focal length
    fl_pix = 400
    int_mat = torch.from_numpy(np.array([[fl_pix * inter_out/1024, 0, inter_out / 2.0], [0, fl_pix * inter_out/1024, inter_out / 2.0], [0, 0, 1.0]], dtype="float32")[None, ...])
    rotmat_world2cam = torch.from_numpy(np.array([[1, 0, 0], [0, np.cos(theta_rot), np.sin(theta_rot)], [0, -np.sin(theta_rot), np.cos(theta_rot)]], dtype="float32"))
    cam_coord = torch.zeros(3,1)

    # Position the camera at the center of the left right axis
    cam_coord[0, 0] = boxes[-1][3] / 2.0

    # How high above the center point do we want to position the camera?
    cam_coord[1, 0] = boxes[-1][4] / 2.0 + min(0.1, abs(boxes[-1][4] / 2.0))

    # Position the camera at the wall close to viewer
    cam_coord[2, 0] = boxes[-1][5]

    # Rotation of the camera (slightly downwards)
    t_world2cam = torch.matmul(rotmat_world2cam, -cam_coord)

    # Conversion between coordinates (from blender to neural mesh)
    # This is different from the pytorch3d conversion...

    # For pytorch3d, instead of matmul use element wise mul***!
    # Use the matrix [[1, -1, 1], [1, 1, -1], [1, -1, 1]]
    # Apply to the output of "get_modelview_matrix" from the keypointnet code
    rotmat_cam2cv = torch.tensor([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]], dtype=torch.float)
    rotmat_world2cv = torch.matmul(rotmat_cam2cv, rotmat_world2cam)
    t_world2cv = torch.matmul(rotmat_cam2cv, t_world2cam)
    return int_mat.cuda(), torch.reshape(rotmat_world2cv, (1, 3, 3)).cuda(), torch.reshape(t_world2cv, (1, 1, 3)).cuda()

def mesh_render_func(boxes, angles, objs, model_ids_old = None, obj_size_target = None):
    b = len(boxes)
    model_ids_return = {}
    obj_size_return = []
    size_loss = 0.0
    # During optimization, we assume that the wall fixed (same as old one)
    old_wall = boxes[-1].clone()
    if not (model_ids_old is None):
        # Overload the room bounding box if optimization iteration != 0
        boxes[-1] = torch.from_numpy(model_ids_old["box_info"]).float().cuda()
    else:
        # Otherwise cache it for future use
        model_ids_return["box_info"] = boxes[-1].detach().cpu().numpy()
    # Retrieve the meshes according to the edge length ratios
    ids = suncg_retrieve(objs, boxes)
    vertices_buf = None
    face_buf = None
    desired_classes = object_idx_to_name[1:][:]
    # The object_idx_to_name variable is loaded by the misc.py file
    desired_classes.append('ceiling')
    desired_classes.append('floor')
    desired_classes.append('wall')
    # We list the classes we want to extract
    model_idx_buffer = {}
    for desired_class in desired_classes:
        model_idx_buffer[desired_class] = []

    valid_obj_offset = 0
    for idx_obj in range(b - 1):
        # Since original boxes were in 0~1 scale, we use the wall max to normalize the boxes
        bbox_min = boxes[idx_obj][:3] * boxes[-1][3:]
        bbox_max = boxes[idx_obj][3:] * boxes[-1][3:]
        obj_center = (bbox_max + bbox_min) / 2
        obj_size = bbox_max - bbox_min
        model_type = object_idx_to_name[objs[idx_obj]]

        model_id = ids[idx_obj]
        if not (model_ids_old is None):
            model_id = model_ids_old[idx_obj]
        else:
            # Cache the retrieved mesh, so it does not change
            model_ids_return[idx_obj] = model_id

        if model_type in ["wall", "ceiling", "floor", "person", "door", "window", "curtain", "blinds"]:
            print("Skipping invalid object")
            # Skip certain non-furniture elements
            # We deal with wall/ceiling/floor later
            continue
        if not (obj_size_target is None):
            # Penalize changes from the original size
            size_loss += nn.functional.mse_loss(obj_size, torch.from_numpy(obj_size_target[valid_obj_offset]).float().cuda())
        else:
            # If iteration is 0, then cache the size
            obj_size_return.append(obj_size.detach().cpu().numpy())
        valid_obj_offset += 1
        theta = -angles[idx_obj] * (2 * float(np.pi) / 24)

        # Probably more efficient to built a dictionary?
        for obj_model in suncg_data[model_type]:
            if obj_model["id"] == model_id:
                model_bbox_min = np.array(obj_model["bbox_min"], dtype=np.float32)
                model_bbox_max = np.array(obj_model["bbox_max"], dtype=np.float32)
                model_size = model_bbox_max - model_bbox_min
                model_size = model_size.astype("float32")
                model_size = torch.from_numpy(model_size).cuda()
                model_center = (model_bbox_min + model_bbox_max) / 2.0
                model_center = model_center.astype("float32")
                model_center = torch.from_numpy(model_center).cuda()
        scale = min([obj_size[0] / model_size[0], obj_size[1] / model_size[1], obj_size[2] / model_size[2]])
        rot = torch.eye(3, dtype=torch.float).cuda()
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rot[0,0] = cos_theta
        rot[0,2] = sin_theta
        rot[2,0] = -sin_theta
        rot[2,2] = cos_theta
        # Compute the translation of the obj
        trans = obj_center - scale * torch.matmul(rot, model_center)
        trans_4x4 = torch.eye(4, dtype=torch.float)
        trans_4x4[:3, -1] = trans
        rot_4x4 = torch.eye(4, dtype=torch.float)
        rot_4x4[:3, :3] = rot * scale
        final_transform_obj = torch.matmul(trans_4x4, rot_4x4)[:3]
        vertices_obj, faces_obj = load_suncg_obj(model_id)
        vertices_obj = torch.t(vertices_obj)
        # make it so we have 3,N as shape
        offset_matrix = torch.ones(1, vertices_obj.shape[1], dtype=torch.float).cuda()
        # Append 1 to get the offset
        vertices_obj = torch.cat((vertices_obj, offset_matrix), dim=0)
        vertices_obj = torch.t(torch.matmul(final_transform_obj.cuda(), vertices_obj))[None, :, :]
        faces_obj = faces_obj[None, :, :]
        if faces_obj.shape[1] > 0:
            # This is for later the extraction of class masks
            if not (model_type in model_idx_buffer):
                model_idx_buffer[model_type] = []
            else:
                pass
            if face_buf is None:
                cur_offset = 0
            else:
                cur_offset = face_buf.shape[1]
            model_idx_buffer[model_type].append([cur_offset, cur_offset + faces_obj.shape[1]])
        else:
            print("Loading of mesh failed")

        # Construct the scene, by joining faces & vertices
        if vertices_buf is None:
            vertices_buf = vertices_obj
            face_buf = faces_obj
        else:
            face_buf = torch.cat((face_buf, faces_obj + vertices_buf.shape[1]), dim=1)
            vertices_buf = torch.cat((vertices_buf, vertices_obj), dim=1)

    # Although we override the wall, the network can still "drift"
    # Use the size loss, but add a positional loss in this case
    if not (obj_size_target is None):
        size_loss += nn.functional.mse_loss(old_wall, torch.from_numpy(obj_size_target[-1]).float().cuda())
    else:
        obj_size_return.append(boxes[-1].detach().cpu().numpy())
    try:
        wall_bbox = boxes[-1][3:]
        obj_center = wall_bbox / 2.0
        obj_size = wall_bbox
        wall_data = wall_retrieve(boxes)
        if not (model_ids_old is None):
            wall_data = model_ids_old["wall"]
        else:
            model_ids_return["wall"] = wall_data
        model_bbox_min = np.array(wall_data["wall_bbox_min"], dtype=np.float32)
        model_bbox_max = np.array(wall_data["wall_bbox_max"], dtype=np.float32)
        model_size = model_bbox_max - model_bbox_min
        model_size = model_size.astype("float32")
        model_size = torch.from_numpy(model_size).cuda()
        model_center = (model_bbox_min + model_bbox_max) / 2.0
        model_center = model_center.astype("float32")
        model_center = torch.from_numpy(model_center).cuda()
        scale = max([obj_size[0] / model_size[0], obj_size[1] / model_size[1], obj_size[2] / model_size[2]])
        rot = torch.eye(3, dtype=torch.float).cuda()
        trans = obj_center - scale * torch.matmul(rot, model_center)
        trans_4x4 = torch.eye(4, dtype=torch.float)
        trans_4x4[:3, -1] = trans
        rot_4x4 = torch.eye(4, dtype=torch.float)
        rot_4x4[:3, :3] = rot * scale
        final_transform_wall = torch.matmul(trans_4x4, rot_4x4)[:3]
        vertices_objs_list, faces_objs_list = load_wall_obj_new(wall_data)
        for cur_mesh in range(len(vertices_objs_list)):
            vertices_obj = vertices_objs_list[cur_mesh]
            faces_obj = faces_objs_list[cur_mesh]
            vertices_obj = torch.t(vertices_obj)
            offset_matrix = torch.ones(1, vertices_obj.shape[1], dtype=torch.float).cuda()
            vertices_obj = torch.cat((vertices_obj, offset_matrix), dim=0)
            vertices_obj = torch.t(torch.matmul(final_transform_wall.cuda(), vertices_obj))[None, :, :]
            faces_obj = faces_obj[None, :, :]
            front_back = vertices_obj[:, :, 2]
            face_buf_offset = front_back[:, faces_obj.long()][0]
            # Wall being too close to camera
            invalid1 = torch.max(face_buf_offset) > 0.9 * boxes[-1][5]
            left_right = vertices_obj[:, :, 0]
            face_buf_left_right = left_right[:, faces_obj.long()][0]
            invalid2 = torch.mean(face_buf_left_right)
            # If the wall is to the middle of the camera left right wise
            invalid2 = invalid2>0.1*boxes[-1][3] and invalid2<0.9*boxes[-1][3]
            # Skip the wall
            if invalid1 and invalid2:
                print("Skipping bad wall")
                continue
            model_type = "wall"
            if faces_obj.shape[1] > 0:
                # This is for later the extraction of class masks
                if not (model_type in model_idx_buffer):
                    model_idx_buffer[model_type] = []
                else:
                    pass
                if face_buf is None:
                    cur_offset = 0
                else:
                    cur_offset = face_buf.shape[1]
                model_idx_buffer[model_type].append([cur_offset, cur_offset + faces_obj.shape[1]])
            if vertices_buf is None:
                vertices_buf = vertices_obj
                face_buf = faces_obj
            else:
                face_buf = torch.cat((face_buf, faces_obj + vertices_buf.shape[1]), dim=1)
                vertices_buf = torch.cat((vertices_buf, vertices_obj), dim=1)
    except Exception as e:
        print("Wall errored")
        print(e)
        pass

    try:
        floor_bbox = boxes[-1][3:]
        obj_center = floor_bbox / 2.0
        obj_size = floor_bbox
        floor_data = floor_retrieve(boxes)
        if not (model_ids_old is None):
            floor_data = model_ids_old["floor"]
        else:
            model_ids_return["floor"] = floor_data
        model_bbox_min = np.array(floor_data["floor_bbox_min"], dtype=np.float32)
        model_bbox_max = np.array(floor_data["floor_bbox_max"], dtype=np.float32)
        model_size = model_bbox_max - model_bbox_min
        model_size = model_size.astype("float32")
        model_size = torch.from_numpy(model_size).cuda()
        model_center = (model_bbox_min + model_bbox_max) / 2.0
        model_center = model_center.astype("float32")
        model_center = torch.from_numpy(model_center).cuda()

        scale = max([obj_size[0] / model_size[0], obj_size[2] / model_size[2]])
        rot = torch.eye(3, dtype=torch.float).cuda()
        obj_center[1] = 0
        trans = obj_center - scale * torch.matmul(rot, model_center)
        trans_4x4 = torch.eye(4, dtype=torch.float)
        trans_4x4[:3, -1] = trans
        rot_4x4 = torch.eye(4, dtype=torch.float)
        rot_4x4[:3, :3] = rot * scale
        final_transform_floor = torch.matmul(trans_4x4, rot_4x4)[:3]
        vertices_obj, faces_obj = load_floor_obj(floor_data)
        vertices_obj = torch.t(vertices_obj)
        offset_matrix = torch.ones(1, vertices_obj.shape[1], dtype=torch.float).cuda()
        vertices_obj = torch.cat((vertices_obj, offset_matrix), dim=0)
        vertices_obj = torch.t(torch.matmul(final_transform_floor.cuda(), vertices_obj))[None, :, :]
        faces_obj = faces_obj[None, :, :]
        model_type = "floor"
        if faces_obj.shape[1] > 0:
            # This is for later the extraction of class masks
            if not (model_type in model_idx_buffer):
                model_idx_buffer[model_type] = []
            else:
                pass

            if face_buf is None:
                cur_offset = 0
            else:
                cur_offset = face_buf.shape[1]
            # print("ADDING FLOOR OFFSET", cur_offset, cur_offset+faces_obj.shape[1])
            model_idx_buffer[model_type].append([cur_offset, cur_offset + faces_obj.shape[1]])
        if vertices_buf is None:
            vertices_buf = vertices_obj
            face_buf = faces_obj
        else:
            face_buf = torch.cat((face_buf, faces_obj + vertices_buf.shape[1]), dim=1)
            vertices_buf = torch.cat((vertices_buf, vertices_obj), dim=1)
    except Exception as e:
        print("Floor errored")
        print(e)
        pass

    try:
        vertices_obj, faces_obj = load_ceil_obj(wall_data)
        ceiling_bbox_new = get_bbox(vertices_obj)
        ceiling_bbox = boxes[-1][3:]
        obj_center = ceiling_bbox / 2.0
        obj_size = ceiling_bbox
        model_bbox_min = ceiling_bbox_new[1]
        model_bbox_max = ceiling_bbox_new[0]
        model_size = model_bbox_max - model_bbox_min
        model_center = (model_bbox_min + model_bbox_max) / 2.0
        scale = max([obj_size[0] / model_size[0], obj_size[2] / model_size[2]])
        scaled_size = scale * model_size
        rot = torch.eye(3, dtype=torch.float).cuda()
        obj_center[1] = 0.5 * scaled_size[1] + ceiling_bbox[1]
        trans = obj_center - scale * torch.matmul(rot, model_center)
        trans_4x4 = torch.eye(4, dtype=torch.float)
        trans_4x4[:3, -1] = torch.t(trans)
        rot *= scale
        rot_4x4 = torch.eye(4, dtype=torch.float)
        rot_4x4[:3, :3] = rot
        final_transform_ceil = torch.matmul(trans_4x4, rot_4x4)[:3]
        vertices_obj = torch.t(vertices_obj)
        offset_matrix = torch.ones(1, vertices_obj.shape[1], dtype=torch.float).cuda()
        vertices_obj = torch.cat((vertices_obj, offset_matrix), dim=0)
        vertices_obj = torch.t(torch.matmul(final_transform_ceil.cuda(), vertices_obj))[None, :, :]
        faces_obj = faces_obj[None, :, :]
        model_type = "ceiling"
        if faces_obj.shape[1] > 0:
            # This is for later the extraction of class masks
            if not (model_type in model_idx_buffer):
                model_idx_buffer[model_type] = []
            else:
                pass
            if face_buf is None:
                cur_offset = 0
            else:
                cur_offset = face_buf.shape[1]
            model_idx_buffer[model_type].append([cur_offset, cur_offset + faces_obj.shape[1]])
        if vertices_buf is None:
            vertices_buf = vertices_obj
            face_buf = faces_obj
        else:
            face_buf = torch.cat((face_buf, faces_obj + vertices_buf.shape[1]), dim=1)
            vertices_buf = torch.cat((vertices_buf, vertices_obj), dim=1)
    except Exception as e:
        print("Ceiling errored")
        print(e)
        pass

    intrisic_mat, rot_mat, trans_mat = get_cam_mat(boxes)
    # Start of our culling code
    culling = True
    eps = 0.06
    if culling:
        vertices_buf_new = torch.matmul(vertices_buf, rot_mat.transpose(1, 2)) + trans_mat
        culling_arr = vertices_buf_new[:, :, 2]
        face_buf_offset = culling_arr[:, face_buf.long()][0]
        face_buf_old_shape = face_buf.shape[1]
        # We use this to initialize the texture later
        invalid = torch.any(face_buf_offset < eps, dim=2)
        valid_offset = ~invalid
        face_buf = face_buf[:, valid_offset[0], :].detach()


    renderer = nr.Renderer(camera_mode='projection', image_size=final_out,
                           K=intrisic_mat, R=rot_mat, t=trans_mat,
                           anti_aliasing=False, orig_size=inter_out, near=0.001, light_intensity_ambient=1.0, light_intensity_directional=0.0)
    texture_size = 2
    textures = torch.ones(1, face_buf.shape[1], texture_size, texture_size, texture_size, 3,
                          dtype=torch.float32).cuda()
    # Render depth
    depth_data = renderer(vertices_buf, face_buf, textures, mode='depth')
    depth_data[depth_data > 15] = -1
    # Infinity depth gets set to negative
    one_hot_matrix = torch.zeros(41, final_out, final_out)
    one_hot_matrix = one_hot_matrix.cuda()

    desired_classes = sorted(list(set(desired_classes)))
    desired_classes.remove("wall")
    desired_classes.insert(0, "wall")
    # Remove 3, because we do not optimize wall, ceiling, floor
    depth_hot_matrix = torch.zeros(len(desired_classes)-3, final_out, final_out)
    depth_hot_matrix = depth_hot_matrix.cuda()
    cur_class_idx = 0
    semantic_container = []
    class_depth_active_container = []
    for class_name in desired_classes:
        if not culling:
            textures = torch.zeros(1, face_buf.shape[1], texture_size, texture_size, texture_size, 3,
                                   dtype=torch.float32).cuda()
        else:
            textures_unculled = torch.zeros(1, face_buf_old_shape, texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
        for offsets in model_idx_buffer[class_name]:
            if culling:
                # new_offset_0 = offsets[0] - cum_missing[max(offsets[0] - 1, 0)]
                # new_offset_1 = offsets[1] - cum_missing[offsets[1] - 1]
                # textures[:, int(new_offset_0):int(new_offset_1)] = 1.0
                textures_unculled[:, offsets[0]:offsets[1]] = 1.0
            else:
                textures[:, offsets[0]:offsets[1]] = 1.0
        if culling:
            textures = textures_unculled[:, valid_offset[0], :]
        textures.requires_grad = False
        images = renderer(vertices_buf, face_buf, textures, mode="rgb")
        # outputshape is 1,3, H, W
        image = torch.sum(images, dim=1, keepdim=True)[0] / 3.0
        hard_mask = image.detach() > 0.1
        # This is the class specific mask
        depth_masked = depth_data[hard_mask]
        class_specific_depth = torch.zeros_like(depth_data)
        class_specific_mean = torch.mean(depth_masked)
        # Take mean depth of active region
        if class_name == "wall":
            wall_max = torch.max(depth_data[hard_mask]).detach()
            if torch.isnan(wall_max):
                wall_max = 10.0
        if torch.isnan(class_specific_mean):
            # So if we are generating the TARGET depth
            # Then an object that does NOT exist in the TARGET
            # Should not be in the image we are optimizing in
            # Hence we set it to be far away
            # ==
            # If an object is NOT in the image we are optimizing
            # Then we can set this to whatever, and it will simply not backprop
            class_specific_mean = wall_max
        class_specific_depth[~hard_mask] = class_specific_mean/wall_max
        class_specific_depth[hard_mask] = depth_data[hard_mask]/wall_max
        if not (class_name in ["wall", "floor", "ceiling"]):
            depth_hot_matrix[cur_class_idx] = class_specific_depth

            cur_class_idx += 1
        # Now the shape is N,C=1,H=1000,W=1000

        # image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        mapped_index = nyu_class.index(class_name.replace("_", " "))
        mapped_index = mapped_index + 1
        one_hot_matrix[mapped_index] = image

    one_hot_matrix = one_hot_matrix[1:]
    final = torch.cat((depth_data, one_hot_matrix, depth_hot_matrix), dim=0)[None,:]
    return final, model_ids_return, obj_size_return, size_loss