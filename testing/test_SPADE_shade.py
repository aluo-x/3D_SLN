import os
import numpy as np
import torch
import imageio
imageio.plugins.freeimage.download()
from models.SPADE_related import SPADEGenerator4
from skimage.transform import resize

colorization_model = SPADEGenerator4(semantic_nc=41, target_nc=3, nz=256, ngf=64, norm='spectralspadelayer3x3',crop_size=256, n_up='normal')
colorization_model_weights = torch.load("./checkpoints/latest_net_G_AB.pth")
colorization_model.load_state_dict(colorization_model_weights)
colorization_model.eval()
colorization_model.cuda()
colorization_model.requires_grad = False

def save_color(data, folder_name='./images', prefix='target'):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    orig_data = data[0]
    orig_data = orig_data.detach().cpu().numpy()
    orig_data = (orig_data + 1.0) / 2.0
    orig_data = orig_data.transpose((1, 2, 0))
    orig_data = orig_data * 255.0
    orig_data = orig_data.astype(np.uint8)
    writer = imageio.get_writer(os.path.join(folder_name, prefix + "_color.png"), mode='i')
    writer.append_data(orig_data)
    writer.close()


def colorize_with_spade(num_z, semantic_dir, save_dir, rooms="all"):
    #
    nyu_class = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat',
                 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower_curtain',
                 'box', 'whiteboard', 'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag',
                 'otherstructure', 'otherfurniture', 'otherprop']
    list_of_z = []
    for z_idx in range(num_z):
        list_of_z.append(torch.randn(1, 256).cuda())
    if rooms == "all":
        rooms = [""]
    else:
        pass
    imgs = [os.path.join(semantic_dir, zzz) for zzz in os.listdir(semantic_dir)]
    depths = list(filter(lambda x:"exr" in x, imgs))
    masks = list(filter(lambda x:(not "depth" in x) and (not "orig" in x), imgs))
    for room_idx in range(len(rooms)):
        cur_depth = list(filter(lambda x:str(rooms[room_idx]) in x, depths))
        cur_masks = list(filter(lambda x: str(rooms[room_idx]) in x, masks))
        depth_data = imageio.imread(cur_depth[0])[..., 0]
        depth_data = depth_data - np.min(depth_data)
        depth_max = np.max(depth_data[depth_data<20])
        depth_data = np.clip(depth_data, 0, depth_max)
        depth_data = depth_data / depth_max
        depth_data = (depth_data - 0.5) * 2
        depth_data = depth_data.astype("float32")[None,:]
        buffer = np.zeros((40, 1024, 1024))
        for mask_idx in range(len(cur_masks)):
            cur_class_data = imageio.imread(cur_masks[mask_idx])
            cur_class_name = os.path.basename(cur_masks[mask_idx])
            # print(cur_class_name.split("_")[2])
            cur_class_name = cur_class_name.split(".")[0]
            cur_class_name_split = cur_class_name.split("_")
            if len(cur_class_name_split)==5:
                cur_name = cur_class_name_split[3]+"_"+cur_class_name_split[4]
            else:
                cur_name = cur_class_name_split[3]
            buffer[nyu_class.index(cur_name)] = cur_class_data[...,0]
        buffer = buffer.astype("float32")
        buffer[buffer<120] = 0.0
        buffer[buffer>120] = 1.0
        total = np.vstack([depth_data, buffer])
        total = np.moveaxis(total, 0, 2)
        total = resize(total, [256, 256], preserve_range=True, order=3, anti_aliasing=True)
        total = np.moveaxis(total, 2, 0)[None,:]
        total = torch.from_numpy(total).float().cuda()
        for attemp in range(len(list_of_z)):
            color_z = list_of_z[attemp]
            new_img = colorization_model(total, color_z)
            save_color(new_img, prefix=str(rooms[room_idx])+"_"+str(attemp).zfill(3), folder_name=save_dir)