import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import tensor_aug, load_json, write_json, get_model_attr
from build_dataset_model import build_suncg_dsets, build_model
import pickle
from models.diff_render import mesh_render_func
import imageio
from data.suncg_dataset import suncg_collate_fn
from testing.test_utils import get_eight_coors_bbox_new, get_iou_cuboid
from testing.test_plot2d import plot2d
suncg_valid_types = load_json("metadata/valid_types.json")
do_not_vis = ["wall", "ceiling", "floor", "person", "door", "window", "curtain", "blinds"]
matching_loss_func = torch.nn.L1Loss()
ce_loss_func = torch.nn.CrossEntropyLoss()

def softargmax(input_vec, sum_dim, beta=2.0):
  idx_vector = torch.cumsum(torch.ones_like(input_vec), dim=sum_dim)
  soft_idx = torch.nn.functional.softmax(input_vec*beta, dim=sum_dim)
  soft_idx_new = torch.mul(soft_idx, idx_vector)
  final_idx = torch.sum(soft_idx_new, dim=sum_dim) - 1.0
  return final_idx

def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return True

nyu_class = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator','television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

mapped_colors = [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),
       (247, 182, 210),		# desk
       (66, 188, 102),
       (219, 219, 141),		# curtain
       (140, 57, 197),
       (202, 185, 52),
       (51, 176, 203),
       (200, 54, 131),
       (92, 193, 61),
       (78, 71, 183),
       (172, 114, 82),
       (255, 127, 14), 		# refrigerator
       (91, 163, 138),
       (153, 98, 156),
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209),
       (227, 119, 194),		# bathtub
       (213, 92, 176),
       (94, 106, 211),
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]

def get_boxes(objs, input_boxes_orig, input_rots_orig):
    input_boxes = input_boxes_orig.detach().cpu().float()
    input_rots = input_rots_orig.detach().cpu().float()
    bbox_coors_min = []
    bbox_coors_min_max = []
    bbox_coors_max = []
    bbox_coors_max_min = []
    total_boxes = []
    for obj_idx in range(len(objs)):
        model_type = suncg_valid_types[int(objs[obj_idx])]
        if model_type in do_not_vis:
            continue
        bbox_min = input_boxes[obj_idx][:3] * input_boxes[-1][3:]
        bbox_max = input_boxes[obj_idx][3:] * input_boxes[-1][3:]
        obj_center = (bbox_max + bbox_min) / 2
        bbox_min = bbox_min - obj_center
        bbox_max = bbox_max - obj_center
        bbox_min_max = bbox_min.detach().clone()
        bbox_min_max[2] = bbox_max[2]
        bbox_max_min = bbox_min.detach().clone()
        bbox_max_min[0] = bbox_max[0]
        rot = torch.eye(3, dtype=torch.float)
        theta = -input_rots[obj_idx] * (2.0 * float(np.pi) / 24.0)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rot[0, 0] = cos_theta
        rot[0, 2] = sin_theta
        rot[2, 0] = -sin_theta
        rot[2, 2] = cos_theta
        trans = obj_center.numpy()
        bbox_coors_min.append(torch.matmul(rot, bbox_min).detach().cpu().numpy() + trans)
        bbox_coors_min_max.append(torch.matmul(rot, bbox_min_max).detach().cpu().numpy() + trans)
        bbox_coors_max.append(torch.matmul(rot, bbox_max).detach().cpu().numpy() + trans)
        bbox_coors_max_min.append(torch.matmul(rot, bbox_max_min).detach().cpu().numpy() + trans)
        cur_box = np.array(get_eight_coors_bbox_new(bbox_coors_min[-1], bbox_coors_max[-1],
                                                    bbox_coors_min_max[-1],
                                                    bbox_coors_max_min[-1]))
        total_boxes.append(cur_box)
    return total_boxes

def save_label_depth(data, depth_data, folder_name='./images', prefix='target'):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    orig_data = data[0][0]
    orig_data = orig_data.detach().cpu().numpy()
    final_out = np.zeros((256,256,3))
    for i in range(1, len(nyu_class)+1):
        image = np.copy(orig_data)
        cur_class = np.isclose(image, i - 1)
        # image[~cur_class] = 0
        final_out[cur_class] = mapped_colors[nyu_class.index(nyu_class[i - 1]) +1]
    # Now save the extra empy class
    image = np.copy(orig_data)
    cur_class = image < -1
    final_out[cur_class] = (0,0,0)
    writer = imageio.get_writer(os.path.join(folder_name, prefix + "_{}.png".format("class_color")), mode='i')
    writer.append_data((1.0 * final_out).astype(np.uint8))
    writer.close()
    depth = depth_data[0][0].detach().cpu().numpy()
    depth = depth - np.min(depth)
    depth[depth > 10.0] = np.max(depth[depth < 10.0])
    depth = depth / np.max(depth[depth < 10.0])
    depth = depth * 255.0
    writer = imageio.get_writer(os.path.join(folder_name, prefix + "_depth.png"), mode='i')
    writer.append_data(depth.astype(np.uint8))

def save_images(data,save_semantic=False, folder_name='./images', prefix='target'):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    orig_data = data[0]
    orig_data = orig_data.detach().cpu().numpy()
    depth = orig_data[0]
    writer = imageio.get_writer(os.path.join(folder_name, prefix + "_depth.gif"), mode='I')
    depth = depth - np.min(depth)
    depth[depth>10.0] = np.max(depth[depth<10.0])
    depth = depth/np.max(depth[depth<10.0])
    # print(np.min(depth), np.max(depth), "MIN MAX DEPTH")
    depth = depth * 255.0
    writer.append_data(depth.astype(np.uint8))
    writer.close()
    if save_semantic:
        for i in range(1, orig_data.shape[0]):
            image = orig_data[i]
            writer = imageio.get_writer(os.path.join(folder_name, prefix+"_{}.gif".format(nyu_class[i-1])), mode='I')
            writer.append_data((255 * image).astype(np.uint8))
            writer.close()

def save_label(data, folder_name='./images', prefix='target'):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    orig_data = data[0]
    orig_data = orig_data.detach().cpu().numpy()
    if orig_data.shape[0] == 1:
        print("Activating semantic saver")
        for i in range(1, len(nyu_class)+1):
            image = np.copy(orig_data)
            cur_class = np.isclose(image, i - 1)
            image[cur_class] = 255
            image[~cur_class] = 0
            # print(image.max(), image.min(), i, "MAX,min,curi")
            writer = imageio.get_writer(os.path.join(folder_name, prefix + "_{}.gif".format(nyu_class[i - 1])),mode='I')
            # data_to_write = (1.0 * image).astype(np.uint8)
            # print(data_to_write.shape, np.max(data_to_write), np.min(data_to_write), "shape, max, min")
            writer.append_data((1.0 * image).astype(np.uint8)[0])
            writer.close()
        # Now save the extra empy class
        image = np.copy(orig_data)
        cur_class = image < -1
        image[cur_class] = 255
        image[~cur_class] = 0
        writer = imageio.get_writer(os.path.join(folder_name, prefix + "_{}.gif".format("empty_class")), mode='I')
        writer.append_data((1.0 * image).astype(np.uint8)[0])
        writer.close()

class PSP_pool_new(nn.Module):
    def __init__(self, sizes=(32,48,64,96), use_max=False,output_list=False):
        super().__init__()
        self.max_size = sizes[-1]
        self.semantic_mode = output_list
        self.pool = nn.Upsample
        self.use_max = use_max
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])


    def _make_stage(self, size):
        if not self.use_max:
            prior = self.pool(size=(size, size), mode='bilinear', align_corners=True)
        else:
            prior = nn.AdaptiveMaxPool2d(output_size=(size, size))
        return prior

    def forward(self, feats):
        priors = [F.upsample(input=stage(feats), size=(self.max_size, self.max_size), mode='bilinear') for stage in self.stages]
        if not self.semantic_mode:
            output_cat = torch.cat(priors, 1)
        else:
            output_cat = priors
        return output_cat

depth_pooler = PSP_pool_new()
semantic_pooler_novel = PSP_pool_new(use_max=False,output_list=True)

def fix_grad(grad_val):
    mod_grad = grad_val.clone().detach()
    avg_val = mod_grad[:, 3:] / 2.0 + mod_grad[:, :3] / 2.0
    mod_grad[:, 3:] = avg_val
    mod_grad[:, :3] = avg_val
    return mod_grad

def quad_grad(grad_val):
    mod_grad = grad_val.clone().detach()
    mod_grad = mod_grad * 4.0
    return mod_grad

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
import string

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def finetune_VAE(args, used_ids, all_save_folders):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    vocab, train_dset, val_dset = build_suncg_dsets(args)
    possible_ids = val_dset.return_room_ids()
    num_to_test = len(used_ids)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    for trial in range(num_to_test):
        save_name = all_save_folders[trial]

        print("Processing trial {}".format(trial))
        if not os.path.isdir(save_name):
            os.mkdir(save_name)
        model, model_kwargs = build_model(args, vocab)
        model.type(float_dtype)
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
        print('Restoring from checkpoint:', restore_path)
        checkpoint = torch.load(restore_path)
        get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
        model.eval()
        z = None
        batch = suncg_collate_fn([val_dset.get_by_room_id(used_ids[trial])])
        batch = tensor_aug(batch, volatile=True)
        ids, objs, boxes_gt, triples, angles, attributes, obj_to_img, triple_to_img = batch
        orig_bbox = None
        Niter_train = 60
        model_infos = None
        size_infos = None
        target_mesh = None
        mu, logvar = model.encoder(objs, triples, boxes_gt, angles, attributes)
        torch.manual_seed(13)
        torch.cuda.manual_seed_all(13)
        # Uncomment if needed...
        z_dummy = reparameterize(mu, logvar)
        z_np = z_dummy.detach().cpu().numpy()
        for k in range(Niter_train):
            print("Processing iter {}".format(k))
            if z is None:
                with open(os.path.join(save_name, "z_value.pkl"), "wb") as f:
                    pickle.dump(z_np, f)
                z = torch.from_numpy(z_np).type(float_dtype).detach()
                z.requires_grad = True
            optimizer = torch.optim.SGD([{'params': [z]}, {'params': model.parameters(),'lr':args.learning_rate / 10.0}], lr=2e-4, nesterov=True, momentum=0.1)
            boxes_pred, angles_pred = model.decoder(z, objs, triples, attributes)
            boxes_pred.register_hook(fix_grad)
            # Mostly stop the size from changing
            obj_to_img_cpu = obj_to_img.data.cpu().clone().tolist()
            boxes_pred[-1] = boxes_gt[-1]

            angles_pred_idx2 = softargmax(angles_pred, sum_dim=1) + torch.randn(angles_pred.shape[0]).cuda()/10.0
            # plot2d(boxes_pred.cpu().detach(), angles_pred_idx2.cpu().detach(), objs.cpu().detach().tolist(), "./test.png")
            # plot2d(boxes_gt.cpu().detach(), angles.cpu().detach().float(), objs.cpu().detach().tolist(), "./test2.png")
            # exit()
            angles_pred_idx2.register_hook(quad_grad)
            angles_pred_idx2[-1] = angles[-1]

            objs_cpu = objs.data.cpu().tolist()
            N = max(obj_to_img_cpu) + 1  # No. images
            orig_scaler = 0.5
            for i in range(N):
                objs_img = []
                angles_gt_img = []
                angles_pred_img = []
                boxes_gt_img = []
                boxes_pred_img = []
                for j in range(len(obj_to_img_cpu)):
                    if obj_to_img_cpu[j] == i:
                        objs_img.append(objs_cpu[j])
                        angles_pred_img.append(angles_pred_idx2[j])
                        boxes_pred_img.append(boxes_pred[j])
                        angles_gt_img.append(angles[j].float())
                        boxes_gt_img.append(boxes_gt[j].float())

                if target_mesh is None:
                    print("Rendering gt")
                    target_image, unused, unused_sizes, unused_size_loss = mesh_render_func(boxes_gt_img, angles_gt_img, objs_img)
                    save_images(target_image, prefix="target", folder_name=save_name)
                    target_image_np = target_image.detach().cpu().numpy()

                print("Rendering novel")
                iter_image, model_infos_from_render, size_from_render, size_loss = mesh_render_func(boxes_pred_img, angles_pred_img, objs_img, model_infos, size_infos)
                if model_infos is None:
                    model_infos = model_infos_from_render
                if size_infos is None:
                    size_infos = size_from_render
                target = torch.from_numpy(target_image_np).cuda()

                # Fill in null regions
                iter_image[:, -1][torch.sum(iter_image[:, 41:], dim=1) < 0.5] = 1.0

                scaled_target_depth = depth_pooler(target[:, 41:])
                scaled_input_depth = depth_pooler(iter_image[:, 41:])
                target_labels = target[:,1:41]
                train_labels = iter_image[:, 1:41]
                train_labels_pooled = semantic_pooler_novel(train_labels)
                if target_mesh is None:
                    target_labels_pooled = semantic_pooler_novel(target_labels)
                    target_container = []
                    for pooled_idx in range(len(target_labels_pooled)):
                        flat_target = torch.argmax(target_labels_pooled[pooled_idx], dim=1, keepdim=True)
                        flat_target[torch.sum(target_labels_pooled[pooled_idx], dim=1, keepdim=True)<0.5] = -100
                        target_container.append(flat_target.detach())
                target_mesh = 1
                semantic_loss = 0.0
                for scale_idx in range(len(train_labels_pooled)):
                    semantic_loss += ce_loss_func(train_labels_pooled[scale_idx],(target_container[scale_idx][:, 0, :, :]).type(long_dtype)) / 800.0
                depth_loss = matching_loss_func(scaled_input_depth, scaled_target_depth) * orig_scaler
                print("Depth loss, semantic loss {}, {}".format(depth_loss, semantic_loss))
                loss_val = depth_loss*100 + semantic_loss*100
                if not (size_infos is None):
                    loss_val += size_loss * 2.0
                    print("Size loss {}".format(size_loss))
                print("Current loss: {}".format(loss_val))
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()
                # if print_iou:
                # Kinda slow to run per iteration
                #     all_ious = []
                #     if orig_bbox is None:
                #         orig_bbox = get_boxes(objs, boxes_gt, angles.float())
                #     cur_bbox = get_boxes(objs, boxes_pred, angles_pred_idx2)
                #     for obj_iou_idx in range(len(orig_bbox)):
                #         all_ious.append(get_iou_cuboid(orig_bbox[obj_iou_idx], cur_bbox[obj_iou_idx]))
                #     print("Current IOU: {}".format(np.mean(all_ious)))
                if k==59 or k==0:
                    if k==0:
                        with open(os.path.join(save_name, "bbox_rot_{}.pkl".format(k)), "wb") as f:
                            depth_mse = matching_loss_func(iter_image[:, 41:], target[:, 41:])
                            cross_entropy = ce_loss_func(train_labels_pooled[-1],(target_container[-1][:, 0, :, :]).type(long_dtype))
                            pickle.dump([used_ids[trial],[bp.detach().cpu().numpy() for bp in boxes_pred_img], [ag.detach().cpu().numpy() for ag in angles_pred_img], size_infos, model_infos, depth_mse.item(), cross_entropy.item()], f)
                    with open(os.path.join(save_name, "bbox_rot_gt_{}.pkl".format(k)), "wb") as f:
                        pickle.dump([used_ids[trial],[bp.detach().cpu().numpy() for bp in boxes_gt_img], [ag.detach().cpu().numpy() for ag in angles_gt_img]], f)
                    save_images(iter_image, prefix=str(k).zfill(3), folder_name=save_name)