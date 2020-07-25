from options.options import Options
import os
import torch
from build_dataset_model import build_loaders, build_model
from utils import get_model_attr, calculate_model_losses, tensor_aug
from collections import defaultdict
import math


def main(args):
    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    print(model)
    model.float().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    restore_path = None
    if args.restore_from_checkpoint:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
    if restore_path is not None and os.path.isfile(restore_path):
        print('Restoring from checkpoint:')
        print(restore_path)
        checkpoint = torch.load(restore_path)
        get_model_attr(model, 'load_state_dict')(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        t = checkpoint['counters']['t']
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()
        epoch = checkpoint['counters']['epoch']
    else:
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'vocab': vocab,
            'model_kwargs': model_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None,
            'optim_state': None,
        }
    while True:
        if t >= args.num_iterations:
            break
        epoch += 1
        print('Starting epoch %d' % epoch)

        for batch in train_loader:
            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
            t += 1
            if t%50 ==0:
                print("Currently on batch {}".format(t))
            ids, objs, boxes, triples, angles, attributes, obj_to_img, triple_to_img = tensor_aug(batch)
            model_out = model(objs, triples, boxes, angles, attributes, obj_to_img)
            mu, logvar, boxes_pred, angles_pred = model_out

            if args.KL_linear_decay:
                KL_weight = 10 ** (t // 1e5 - 6)
            else:
                KL_weight = args.KL_loss_weight
            total_loss, losses = calculate_model_losses(args, model, boxes, boxes_pred, angles, angles_pred, mu=mu, logvar=logvar, KL_weight=KL_weight)
            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if t % args.print_every == 0:
                print("On batch {} out of {}".format(t, args.num_iterations))
                for name, val in losses.items():
                    print(' [%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                checkpoint['losses_ts'].append(t)

            if t % args.checkpoint_every == 0:
                checkpoint['model_state'] = get_model_attr(model, 'state_dict')()
                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(args.output_dir, 'latest_%s_with_model.pt' % args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

                if t % args.snapshot_every == 0:
                    snapshot_name = args.checkpoint_name + 'snapshot_%06dK' % (t // 1000)
                    snapshot_path = os.path.join(args.output_dir, snapshot_name)
                    print('Saving snapshot to ', snapshot_path)
                    torch.save(checkpoint, snapshot_path)

                checkpoint_path = os.path.join(args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                key_blacklist = ['model_state', 'optim_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)

if __name__ == '__main__':
    args = Options().parse()
    if (args.output_dir is not None) and (not os.path.isdir(args.output_dir)):
        os.mkdir(args.output_dir)
    if (args.test_dir is not None) and (not os.path.isdir(args.test_dir)):
        os.mkdir(args.test_dir)
    main(args)
