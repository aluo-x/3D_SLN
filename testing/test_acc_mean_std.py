from utils import get_model_attr, tensor_aug
import os
import torch
from build_dataset_model import build_model, build_loaders
import numpy as np
import pickle
from testing.test_utils import random_scene, scene_graph_acc
import torch.nn.functional as F

def get_std(args, Nsample=10):
    float_dtype = torch.cuda.FloatTensor
    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
    print('Restoring from checkpoint:', restore_path)
    # restore_path = "/data/vision/billf/scratch/aluo/3d_SLN/checkpoints/latest_checkpoint_with_model.pt"
    checkpoint = torch.load(restore_path)
    get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
    model.eval()

    print("Loading cached mean & cov")
    args.test_dir = "/data/vision/billf/scratch/aluo/3d_scene_vae"
    stats_file = os.path.join(args.test_dir, "mean_cov_public.pkl")
    if not os.path.isfile(stats_file):
        print("Compute the stats first!")
        print("Exiting...")
        exit()
    with open(stats_file, "rb") as mean_cov_f:
        mean_est, cov_est = pickle.load(mean_cov_f)
        print("Finished loading")
    std_angle = []
    std_position = []
    std_size = []
    i = 0
    with torch.no_grad():
        for batch in val_loader:
            i += 1
            print("Evaluating {} out of {}".format(i, len(val_loader)))
            boxes_batch = []
            angles_batch = []
            positions_batch = []
            sizes_batch = []
            batch = tensor_aug(batch)
            for k in range(Nsample):
                ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = batch
                z_est = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float()
                z_est = z_est.type(float_dtype).cuda()
                boxes_pred, angles_pred = model.decoder(z_est, objs, triples, attributes)
                boxes_pred = boxes_pred.data.cpu().numpy()
                prob, angles_pred = angles_pred.data.cpu().clone().max(1)
                boxes_batch.append(boxes_pred)
                angles_batch.append(angles_pred.numpy())
                positions_batch.append(boxes_pred[:, :3] / 2.0 + boxes_pred[:, 3:] / 2.0)
                sizes_batch.append(np.abs(boxes_pred[:, :3] - boxes_pred[:, 3:]))
            angles_col = np.stack(angles_batch, axis=0)
            positions_col = np.stack(positions_batch, axis=0)
            sizes_col = np.stack(sizes_batch, axis=0)
            angle_std = np.mean(np.std(angles_col, axis=0))
            positions_std = np.mean(np.std(positions_col, axis=0))
            sizes_std = np.mean(np.std(sizes_col, axis=0))
            std_angle.append(angle_std)
            std_position.append(positions_std)
            std_size.append(sizes_std)
        print("mean angle std:", np.mean(std_angle))
        print("mean pos std:", np.mean(std_position))
        print("mean sizes std:", np.mean(std_size))

def get_acc_l1(args):
    float_dtype = torch.cuda.FloatTensor
    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
    print('Restoring from checkpoint:', restore_path)
    # restore_path = "/data/vision/billf/scratch/aluo/3d_SLN/checkpoints/latest_checkpoint_with_model.pt"
    checkpoint = torch.load(restore_path)
    get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
    model.eval()
    print("Loading cached mean & cov")
    args.test_dir = "/data/vision/billf/scratch/aluo/3d_scene_vae"
    stats_file = os.path.join(args.test_dir, "mean_cov_public.pkl")
    if not os.path.isfile(stats_file):
        print("Compute the stats first!")
        print("Exiting...")
        exit()
    with open(stats_file, "rb") as mean_cov_f:
        mean_est, cov_est = pickle.load(mean_cov_f)
        print("Finished loading")

    acc_pred = 0
    acc_rand = 0
    acc_pert = 0
    total_triple = 0
    l1_loss_pred = []
    l1_loss_rand = []
    l1_loss_pert = []
    i=0
    with torch.no_grad():
        for batch in val_loader:
            i+=1
            print("Evaluating {} out of {}".format(i, len(val_loader)))
            batch = tensor_aug(batch)
            ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = batch
            z_est = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float()
            z_est = z_est.type(float_dtype)
            boxes_pred_recon, angles_pred_recon = model.decoder(z_est, objs, triples, attributes)
            boxes_pred_rand, angles_pred_recond = random_scene(objs, boxes, angles)
            boxes_offset = np.random.normal(0, 0.1, [boxes.size(0), 3])
            boxes_pred_perturb = torch.from_numpy(np.hstack([boxes_offset, boxes_offset])).type(float_dtype) + boxes
            l1_loss_pred.append(F.l1_loss(boxes_pred_recon.cpu(), boxes.cpu()).item())
            l1_loss_rand.append(F.l1_loss(boxes_pred_rand.cpu(), boxes.cpu()).item())
            l1_loss_pert.append(F.l1_loss(boxes_pred_perturb.cpu(), boxes.cpu()).item())
            acc_pred += scene_graph_acc(vocab, objs.data, triples.data, boxes_pred_recon.data)
            acc_rand += scene_graph_acc(vocab, objs.data, triples.data, boxes_pred_rand.data)
            acc_pert += scene_graph_acc(vocab, objs.data, triples.data, boxes_pred_perturb.data)
            total_triple += triples.size(0)
    total_triple = float(total_triple)
    print("PRED, RAND, PERT L1:", np.mean(l1_loss_pred), np.mean(l1_loss_rand), np.mean(l1_loss_pert))
    print("PRED, RAND, PERT ACC: ", acc_pred/total_triple, acc_rand/total_triple, acc_pert/total_triple)
    return 1
