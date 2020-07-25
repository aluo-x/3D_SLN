from data.suncg_dataset import SuncgDataset, suncg_collate_fn
from torch.utils.data import DataLoader
import json
from models.Sg2ScVAE_model import Sg2ScVAEModel


def build_suncg_dsets(args):
    dset_kwargs = {
        'data_dir': args.suncg_train_dir,
        'train_3d': args.train_3d,
        'use_attr_30': args.use_attr_30,
    }
    train_dset = SuncgDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)
    print('Training dataset has %d scenes and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
    dset_kwargs['data_dir'] = args.suncg_val_dir
    val_dset = SuncgDataset(**dset_kwargs)
    assert train_dset.vocab == val_dset.vocab
    vocab = json.loads(json.dumps(train_dset.vocab))
    return vocab, train_dset, val_dset


def build_loaders(args):
    vocab, train_dset, val_dset = build_suncg_dsets(args)
    collate_fn = suncg_collate_fn
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
        'collate_fn': collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    loader_kwargs['shuffle'] = False
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader

def build_model(args, vocab):
    kwargs = {
        'vocab': vocab,
        'batch_size': args.batch_size,
        'train_3d': args.train_3d,
        'decoder_cat': args.decoder_cat,
        'embedding_dim': args.embedding_dim,
        'gconv_mode': args.gconv_mode,
        'gconv_num_layers': args.gconv_num_layers,
        'mlp_normalization': args.mlp_normalization,
        'vec_noise_dim': args.vec_noise_dim,
        'layout_noise_dim': args.layout_noise_dim,
        'use_AE': args.use_AE
    }
    model = Sg2ScVAEModel(**kwargs)
    if args.multigpu:
        assert False, 'Multi-GPU not supported'
    return model, kwargs
