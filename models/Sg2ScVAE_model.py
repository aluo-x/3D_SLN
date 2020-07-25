import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph import make_mlp, GraphTripleConvNet, _init_weights

class Sg2ScVAEModel(nn.Module):
    def __init__(self, vocab, embedding_dim=128, batch_size=32,
                 train_3d=True,
                 decoder_cat=False,
                 Nangle=24,
                 gconv_mode='feedforward',
                 gconv_pooling='avg', gconv_num_layers=5,
                 mlp_normalization='none',
                 vec_noise_dim=0,
                 layout_noise_dim=0,
                 use_AE=False,
                 use_attr=True):
        super(Sg2ScVAEModel, self).__init__()
        gconv_dim = embedding_dim
        gconv_hidden_dim = gconv_dim * 4
        box_embedding_dim = int(embedding_dim * 3 / 4)
        angle_embedding_dim = int(embedding_dim / 4)
        attr_embedding_dim = 0
        obj_embedding_dim = embedding_dim

        self.use_attr = use_attr
        self.batch_size = batch_size
        self.train_3d = train_3d
        self.decoder_cat = decoder_cat
        self.vocab = vocab
        self.vec_noise_dim = vec_noise_dim
        self.layout_noise_dim = layout_noise_dim
        self.use_AE = use_AE

        if self.use_attr:
            obj_embedding_dim = int(embedding_dim * 3 / 4)
            attr_embedding_dim = int(embedding_dim / 4)

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        num_attrs = len(vocab['attrib_idx_to_name'])

        # making nets
        self.obj_embeddings_ec = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_ec = nn.Embedding(num_preds, embedding_dim * 2)
        self.obj_embeddings_dc = nn.Embedding(num_objs + 1, obj_embedding_dim)
        self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim)
        if use_attr:
            self.attr_embedding_ec = nn.Embedding(num_attrs, attr_embedding_dim)
            self.attr_embedding_dc = nn.Embedding(num_attrs, attr_embedding_dim)
        if self.decoder_cat:
            self.pred_embeddings_dc = nn.Embedding(num_preds, embedding_dim * 2)
        if self.train_3d:
            self.box_embeddings = nn.Linear(6, box_embedding_dim)
        else:
            self.box_embeddings = nn.Linear(4, box_embedding_dim)
        self.angle_embeddings = nn.Embedding(Nangle, angle_embedding_dim)
        # weight sharing of mean and var
        self.box_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                     batch_norm=mlp_normalization)
        self.box_mean = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.box_var = make_mlp([embedding_dim * 2, box_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.angle_mean_var = make_mlp([embedding_dim * 2, gconv_hidden_dim, embedding_dim * 2],
                                       batch_norm=mlp_normalization)
        self.angle_mean = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)
        self.angle_var = make_mlp([embedding_dim * 2, angle_embedding_dim], batch_norm=mlp_normalization, norelu=True)        # graph conv net
        self.gconv_net_ec = None
        self.gconv_net_dc = None
        if gconv_num_layers > 0:
            gconv_kwargs_ec = {
                'input_dim': gconv_dim * 2,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mode': gconv_mode,
                'mlp_normalization': mlp_normalization,
            }
            gconv_kwargs_dc = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'pooling': gconv_pooling,
                'num_layers': gconv_num_layers,
                'mode': gconv_mode,
                'mlp_normalization': mlp_normalization,
            }
            self.gconv_net_ec = GraphTripleConvNet(**gconv_kwargs_ec)
            if self.decoder_cat:
                gconv_kwargs_dc['input_dim'] = gconv_dim * 2
            self.gconv_net_dc = GraphTripleConvNet(**gconv_kwargs_dc)

        # box prediction net
        if self.train_3d:
            box_net_dim = 6
        else:
            box_net_dim = 4
        box_net_layers = [gconv_dim * 2, gconv_hidden_dim, box_net_dim]
        if self.use_attr:
            box_net_layers = [gconv_dim * 2 + attr_embedding_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = make_mlp(box_net_layers, batch_norm=mlp_normalization, norelu=True)

        # angle prediction net
        angle_net_layers = [gconv_dim * 2, gconv_hidden_dim, Nangle]
        self.angle_net = make_mlp(angle_net_layers, batch_norm=mlp_normalization, norelu=True)

        # initialization
        self.box_embeddings.apply(_init_weights)
        self.box_mean_var.apply(_init_weights)
        self.box_mean.apply(_init_weights)
        self.box_var.apply(_init_weights)
        self.angle_mean_var.apply(_init_weights)
        self.angle_mean.apply(_init_weights)
        self.angle_var.apply(_init_weights)
        self.box_net.apply(_init_weights)

    def encoder(self, objs, triples, boxes_gt, angles_gt, attributes):
        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_ec(objs)
        if self.use_attr:
            attr_vecs = self.attr_embedding_ec(attributes)
            obj_vecs = torch.cat([obj_vecs, attr_vecs], dim=1)
        angle_vecs = self.angle_embeddings(angles_gt)
        pred_vecs = self.pred_embeddings_ec(p)
        boxes_vecs = self.box_embeddings(boxes_gt)

        obj_vecs = torch.cat([obj_vecs, boxes_vecs, angle_vecs], dim=1)

        if self.gconv_net_ec is not None:
            obj_vecs, pred_vecs = self.gconv_net_ec(obj_vecs, pred_vecs, edges)

        obj_vecs_box = self.box_mean_var(obj_vecs)
        mu_box = self.box_mean(obj_vecs_box)
        logvar_box = self.box_var(obj_vecs_box)

        obj_vecs_angle = self.angle_mean_var(obj_vecs)
        mu_angle = self.angle_mean(obj_vecs_angle)
        logvar_angle = self.angle_var(obj_vecs_angle)
        mu = torch.cat([mu_box, mu_angle], dim=1)
        logvar = torch.cat([logvar_box, logvar_angle], dim=1)
        return mu, logvar

    def decoder(self, z, objs, triples, attributes):
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        obj_vecs = self.obj_embeddings_dc(objs)
        if self.use_attr:
            attr_vecs = self.attr_embedding_dc(attributes)
            obj_vecs = torch.cat([obj_vecs, attr_vecs], dim=1)
        pred_vecs = self.pred_embeddings_dc(p)

        # concatenate noise first
        if self.decoder_cat:
            obj_vecs = torch.cat([obj_vecs, z], dim=1)
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)

        # concatenate noise after gconv
        else:
            obj_vecs, pred_vecs = self.gconv_net_dc(obj_vecs, pred_vecs, edges)
            obj_vecs = torch.cat([obj_vecs, z], dim=1)

        if self.use_attr:
            obj_vecs_box = torch.cat([obj_vecs, attr_vecs], dim=1)
            boxes_pred = self.box_net(obj_vecs_box)
        else:
            boxes_pred = self.box_net(obj_vecs)
        angles_pred = F.log_softmax(self.angle_net(obj_vecs), dim=1)
        return boxes_pred, angles_pred

    def forward(self, objs, triples, boxes_gt, angles_gt, attributes, obj_to_img):
        mu, logvar = self.encoder(objs, triples, boxes_gt, angles_gt, attributes)
        if self.use_AE:
            z = mu
        else:
            # reparameterization
            std = torch.exp(0.5*logvar)
            # standard sampling
            eps = torch.randn_like(std)
            z = eps.mul(std).add_(mu)

            # You can theoretically also just sample one per room, instead of one per obj
            # Just replicate it
        boxes_pred, angles_pred = self.decoder(z, objs, triples, attributes)
        return mu, logvar, boxes_pred, angles_pred
