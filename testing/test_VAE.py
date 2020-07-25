from utils import write_json, get_model_attr, tensor_aug
import os
import torch
from build_dataset_model import build_loaders, build_model
import numpy as np
import pickle

# This function generates lots of layouts using our network
def get_layouts_from_network(args):
    with torch.no_grad():
        print(args)
        test_data_dir = os.path.join(args.test_dir, "data")
        if not os.path.isdir(test_data_dir):
            os.mkdir(test_data_dir)
        float_dtype = torch.cuda.FloatTensor
        vocab, train_loader, val_loader = build_loaders(args)
        # write_json(os.path.join(args.test_dir, "vocab.json"), vocab)
        model, model_kwargs = build_model(args, vocab)
        model.type(float_dtype)
        print(model)
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
        print('Restoring from checkpoint: {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
        model.eval()
        print("getting mean and covariance from training data")
        mean_cat = None
        print("Starting to collect data")
        total_count = len(train_loader)
        cur_iter = 1
        mean_cov_path = os.path.join(args.test_dir, "mean_cov.pkl")
        if not os.path.isfile(mean_cov_path):
            for batch in train_loader:
                print("returning {}/{}".format(cur_iter, total_count))
                cur_iter += 1
                batch = tensor_aug(batch, volatile=True)
                ids, objs, boxes_gt, triples, angles_gt, attributes, obj_to_img, triple_to_img = batch
                mean, logvar = model.encoder(objs, triples, boxes_gt, angles_gt, attributes)
                mean = mean.data.cpu().clone()
                if mean_cat is None:
                    mean_cat = mean
                else:
                    mean_cat = torch.cat([mean_cat, mean], dim=0)
            print("Finished collecting data")
            mean_est = torch.mean(mean_cat, dim=0, keepdim=True)  # size 1*embed_dim
            mean_cat = mean_cat - mean_est
            n = mean_cat.size(0)
            d = mean_cat.size(1)
            cov_est = np.zeros((d, d))
            for i in range(n):
                x = mean_cat[i].numpy()
                cov_est += 1.0 / (n - 1.0) * np.outer(x, x)
            mean_est = mean_est.numpy()[0]
            print("Start to cache the mean & cov")
            with open(mean_cov_path, "wb") as mean_cov_f:
                pickle.dump([mean_est, cov_est], mean_cov_f)
        else:
            print("Loading cached mean & cov")
            with open(mean_cov_path, "rb") as mean_cov_f:
                mean_est, cov_est = pickle.load(mean_cov_f)
                print("Finished loading")
        data = {}
        total_val_count = len(val_loader)
        cur_val_iter = 1
        for batch in val_loader:
            print("Starting batch {}/{}".format(cur_val_iter, total_val_count))
            cur_val_iter += 1
            batch = tensor_aug(batch, volatile=True)
            ids, objs, boxes_gt, triples, angles, attributes, obj_to_img, triple_to_img = batch
            Nsample = 4
            ids_cpu = ids.data.cpu().clone().tolist()
            objs_cpu = objs.data.cpu().clone().tolist()
            angles_gt_cpu = angles.data.cpu().clone().tolist()
            boxes_gt_cpu = boxes_gt.data.cpu().clone().tolist()
            triples_cpu = triples.data.cpu().clone().tolist()
            obj_to_img_cpu = obj_to_img.data.cpu().clone().tolist()
            triple_to_img_cpu = triple_to_img.data.cpu().clone().tolist()
            for img_id in ids_cpu:
                data[img_id] = {}
            for k in range(Nsample):
                z = torch.from_numpy(np.random.multivariate_normal(mean_est, cov_est, objs.size(0))).float().cuda()
                boxes_pred, angles_pred = model.decoder(z, objs, triples, attributes)
                prob, angles_pred_cpu = angles_pred.data.cpu().clone().max(1)
                angles_pred_cpu = angles_pred_cpu.tolist()
                boxes_pred_cpu = boxes_pred.data.cpu().clone().tolist()
                N = max(obj_to_img_cpu) + 1  # No. images
                Offset = 0
                for i in range(N):
                    objs_img = []
                    angles_gt_img = []
                    angles_pred_img = []
                    boxes_gt_img = []
                    boxes_pred_img = []
                    triples_img = []
                    for j in range(len(obj_to_img_cpu)):
                        if obj_to_img_cpu[j] == i:
                            objs_img.append(objs_cpu[j])
                            angles_gt_img.append(angles_gt_cpu[j])
                            angles_pred_img.append(angles_pred_cpu[j])
                            boxes_gt_img.append(boxes_gt_cpu[j])
                            boxes_pred_img.append(boxes_pred_cpu[j])
                    for jk in range(len(triple_to_img_cpu)):
                        if triple_to_img_cpu[jk] == i:
                            triples_cpu[jk][0] -= Offset
                            triples_cpu[jk][2] -= Offset
                            triples_img.append(triples_cpu[jk])
                    Offset += len(objs_img)
                    data[ids_cpu[i]]["gt"] = {'objs': objs_img,
                                              'angles': angles_gt_img,
                                              'boxes': boxes_gt_img,
                                              'triples': triples_img
                                              }
                    data[ids_cpu[i]][str(k)] = {
                        'angles': angles_pred_img,
                        'boxes': boxes_pred_img,
                    }
        print("Writing extracted json to disk")
        write_json(os.path.join(test_data_dir, "data_extracted.json"), data)
