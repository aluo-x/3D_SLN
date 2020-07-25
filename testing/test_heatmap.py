from utils import get_model_attr
import os
import torch
from build_dataset_model import build_model, build_suncg_dsets
import numpy as np
import pickle
from testing.test_utils import get_sg_from_words
import matplotlib.pyplot as plt

def produce_heatmap(args, objs_in_room=None, rels_in_room=None, num_iter=20000):
    test_data_dir = os.path.join(args.test_dir, "data")
    if not os.path.isdir(test_data_dir):
        os.mkdir(test_data_dir)

    heat_dir = os.path.join(test_data_dir, "heat")
    if not os.path.isdir(heat_dir):
        os.mkdir(heat_dir)

    float_dtype = torch.cuda.FloatTensor
    vocab, train_dset, val_dset = build_suncg_dsets(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
    print('Restoring from checkpoint:', restore_path)
    checkpoint = torch.load(restore_path)
    get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
    model.eval()

    print("Loading cached mean & cov")
    stats_file = os.path.join(args.test_dir, "mean_cov.pkl")
    if not os.path.isfile(stats_file):
        print("Compute the stats first!")
        print("Exiting...")
        exit()
    with open(stats_file, "rb") as mean_cov_f:
        mean_est, cov_est = pickle.load(mean_cov_f)
        print("Finished loading")

    objs5 = ["bed", "desk", "cabinet", "chair", "lamp"]
    rels5 = [("bed", "behind", "desk"), ("cabinet", "left of", "bed"), ("chair", "left of", "desk"),
             ("lamp", "on", "desk")]

    if objs_in_room is None:
        obj_list = [objs5]
        rel_list = [rels5]
    else:
        obj_list = objs_in_room
        rel_list = rels_in_room

    for room_idx in range(len(obj_list)):
        boxes_list = []
        ag_list = []
        with torch.no_grad():
            for k in range(num_iter):
                print("Processing iter {}".format(k))
                objs, triples, attributes = get_sg_from_words(obj_list[room_idx], rel_list[room_idx])
                z_np = np.random.multivariate_normal(mean_est, cov_est, objs.size(0))
                z = torch.from_numpy(z_np).type(float_dtype).detach()
                boxes_pred, angles_pred = model.decoder(z, objs.cuda(), triples.cuda(), attributes.cuda())
                boxes_list.append([bp.detach().cpu().numpy() for bp in boxes_pred])
            with open(os.path.join(heat_dir, str(room_idx).zfill(4)+"_heat.pkl"), "wb") as f:
                pickle.dump([objs.cpu().numpy(),attributes.cpu().numpy(), boxes_list, ag_list], f)

def plot_heatmap(heat_pkl_path, save_dir, visualize=False, clip_coor=True):
    print("Loading {}".format(heat_pkl_path))
    with open(heat_pkl_path, "rb") as f:
        heat_pkl = pickle.load(f)
    heat_pkl_idx = os.path.basename(heat_pkl_path).split("_")[0]
    container_size = 100
    # num of pixels
    print("Found {} trials".format(len(heat_pkl[2])))
    for obj_type in range(len(heat_pkl[2][0]) - 1):
        print("Plotting object {}".format(obj_type))
        container = np.zeros((container_size, container_size)).astype("float")
        for trial in range(len(heat_pkl[2])):
            bbox_new = heat_pkl[2][trial][obj_type]
            normliz = heat_pkl[2][trial][-1]
            bbox_new = np.array(bbox_new) * np.concatenate([normliz[3:]-normliz[:3], normliz[3:]-normliz[:3]])
            ct = (bbox_new[:3] + bbox_new[3:]) * 0.5
            # Either rejection or clip it
            if clip_coor:
                ct = np.clip(ct, 0.0, 1.0)
            else:
                if not (np.all(ct>0.0) and np.all(ct<1.0)):
                    continue
            # Depends on how you want to use the bounding boxes
            rd = np.floor(ct * (container_size-1)).astype("int")
            container[rd[2], rd[0]] += 1.0
        container = container / max(np.sum(container), 1.0)
        plt.imshow(container, cmap="plasma")
        plt.tight_layout()
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        if visualize:
            plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(heat_pkl_idx, str(obj_type).zfill(2))))
            plt.show()
        else:
            plt.savefig(os.path.join(save_dir, '{}_{}.png'.format(heat_pkl_idx, str(obj_type).zfill(2))))
            plt.close()
    exit()
