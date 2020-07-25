import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
import re
# TODO: Use the default SPADE code, and/or release the SPADE training code

def padded_conv(in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
    sequence = []
    sequence += [nn.ReflectionPad2d(padding)]
    sequence += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0,dilation=dilation, groups=groups, bias=bias)]
    return nn.Sequential(*sequence)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', use_bias=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        elif norm == 'spectral':
            self.norm = None
            self.conv = spectral_norm(self.conv)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution


    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class SEBlock2(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResBlock2(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='reflect', nz=0):
        super(SEResBlock2, self).__init__()

        model = []
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        model += [SEBlock2(dim+nz, reduction=4)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class SEResBlock3(nn.Module):
    def __init__(self, inplane, outplane,stride=1, norm='spectral', pad_type='reflect'):
        super(SEResBlock3, self).__init__()

        model = []
        model += [Conv2dBlock(inplane, outplane, 3, stride, 1, norm=norm, activation='lrelu', pad_type=pad_type, use_bias=True)]
        model += [Conv2dBlock(outplane, outplane, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_bias=True)]
        model += [SEBlock2(outplane, reduction=4)]
        self.model = nn.Sequential(*model)
        if (outplane != inplane) or (stride!=1):
            self.learned_skip = Conv2dBlock(inplane, outplane, 3, stride, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        else:
            self.learned_skip = None
        # TODO: REMOVE INPLACE
        self.final_act = nn.LeakyReLU(0.2, inplace=False)
    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.learned_skip is None:
            out += residual
        else:
            out += self.learned_skip(residual)
        final_out = self.final_act(out)
        return final_out

class LayerNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class SPADEGenerator(nn.Module):
    def __init__(self, semantic_nc, target_nc, nz, ngf, norm, crop_size, n_up):
        super().__init__()
        # self.opt = opt
        nf = ngf
        self.nf = ngf
        self.n_up = n_up
        self.sw, self.sh = self.compute_latent_vector_size(n_up, crop_size)
        self.has_z = nz>0
        self.nz = nz
        if self.has_z:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.nz, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, norm, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, norm, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, norm, semantic_nc)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, norm, semantic_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, norm, semantic_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, norm, semantic_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, norm, semantic_nc)

        final_nc = nf

        if n_up == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, norm,semantic_nc)
            final_nc = nf // 2

        self.conv_img_pre = SEResBlock2(final_nc)
        self.conv_img = nn.Conv2d(final_nc, target_nc, 5, padding=2)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, n_up, crop_size):
        if n_up == 'normal':
            num_up_layers = 5
        elif n_up == 'more':
            num_up_layers = 6
        elif n_up == 'most':
            num_up_layers = 7

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             n_up)

        sw = crop_size // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.has_z:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                print("Missing z vector, sampling from normal")
                z = torch.randn(input.size(0), self.nz,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.nf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        seg_1 = F.interpolate(seg, size=[self.sh, self.sw])
        x = self.head_0(x, seg_1)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.n_up == 'more' or \
           self.n_up == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.n_up == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img_pre(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.semantic_nc = semantic_nc
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            raise ValueError
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

def get_nonspade_norm_layer(norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            old_padding = 0
            if layer.padding != (0,0):
                old_padding = layer.padding[0]
                layer.padding = (0,0)
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            # norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
            raise ValueError
            # Did not import the submodule containing syncbatch norm

        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'layer':
            norm_layer = LayerNorm2D(get_out_channel(layer), affine=True)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        padding_layer = None
        if old_padding != 0:
            padding_layer = nn.ReflectionPad2d(old_padding)
        return nn.Sequential(padding_layer, layer, norm_layer)

    return add_norm_layer


# From SPADE
class MultiscaleDiscriminator(nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--netD_subarch', type=str, default='n_layer',
    #                         help='architecture of each discriminator')
    #     parser.add_argument('--num_D', type=int, default=2,
    #                         help='number of discriminators to be used in multiscale')
    #     opt, _ = parser.parse_known_args()
    #
    #     # define properties of each discriminator of the multiscale discriminator
    #     subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
    #                                         'models.networks.discriminator')
    #     subnetD.modify_commandline_options(parser, is_train)
    #
    #     return parser

    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, num_D=2, use_feat_loss=True):
        super().__init__()
        self.use_feat_loss = use_feat_loss
        for i in range(num_D):
            subnetD = self.create_single_discriminator(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
            self.add_module('discriminator_%d' % i, subnetD)
            n_layers = n_layers - 1

    def create_single_discriminator(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            print("Selected n_layer pix2pixHD discrim")
            netD = NLayerDiscriminator(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.use_feat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# From SPADE
class NLayerDiscriminator(nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--n_layers_D', type=int, default=3,
    #                         help='# layers in each discriminator')
    #     return parser

    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss):
        super().__init__()
        # self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf
        if conditional_nc <= 0:
            print("Creating Pix2PixHD discriminator")
            print("0 dimensional input set")
        input_nc_total = input_nc + conditional_nc

        norm_layer = get_nonspade_norm_layer(norm_layer)
        sequence = [[nn.Conv2d(input_nc_total, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]
        self.use_feat_loss = use_feat_loss
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride_val = 1 if n == n_layers - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride_val, padding=padw)),
                          nn.LeakyReLU(0.2, True)
                          ]]

        # sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=1)]]
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    # def compute_D_input_nc(self, opt):
    #     input_nc = opt.label_nc + opt.output_nc
    #     if opt.contain_dontcare_label:
    #         input_nc += 1
    #     if not opt.no_instance:
    #         input_nc += 1
    #     return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = self.use_feat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

# Version of GANLoss from SPADE
class GANLoss_2(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss_2, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls' or gan_mode == 'lsgan':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls' or self.gan_mode == 'lsgan':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                if len(pred_i) == 2:
                    pred_i = pred_i[0]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, input_nc, output_nc, nef, norm_layer_str, crop_size):
        super().__init__()
        self.crop_size = crop_size
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        # ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(norm_layer_str)
        self.layer1 = norm_layer(nn.Conv2d(input_nc, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef * 1, nef * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef * 2, nef * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(nef * 4, nef * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        if self.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))

        # s0 = crop_size//2**5
        # if self.crop_size >= 256:
        #     s0 = s0//2


        self.fc_mu = nn.Linear(nef * 8, output_nc)
        self.fc_var = nn.Linear(nef * 8, output_nc)
        self.actvn = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.crop_size >= 256:
            x = self.layer6(self.actvn(x))

        x = self.pool_layer(x)
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

class SPADEGenerator2(nn.Module):
    def __init__(self, semantic_nc, target_nc, nz, ngf, norm, crop_size, n_up):
        super().__init__()
        # self.opt = opt
        nf = ngf
        self.nf = ngf
        self.n_up = n_up
        self.sw, self.sh = self.compute_latent_vector_size(n_up, crop_size)
        self.has_z = nz>0
        self.nz = nz
        # todo: replace 8 with 16
        if self.has_z:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.nz, 12 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 12 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock2(12 * nf, 12 * nf, norm, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock2(12 * nf, 12 * nf, norm, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock2(12 * nf, 12 * nf, norm, semantic_nc)

        self.up_0 = SPADEResnetBlock2(12 * nf, 8 * nf, norm, semantic_nc)
        self.up_1 = SPADEResnetBlock2(8 * nf, 4 * nf, norm, semantic_nc)
        self.up_2 = SPADEResnetBlock2(4 * nf, 2 * nf, norm, semantic_nc)
        self.up_3 = SPADEResnetBlock2(2 * nf, 1 * nf, norm, semantic_nc)

        final_nc = nf

        if n_up == 'most':
            self.up_4 = SPADEResnetBlock2(1 * nf, nf // 2, norm,semantic_nc)
            final_nc = nf // 2

        self.conv_img_pre = SEResBlock2(final_nc)
        self.conv_img = nn.Conv2d(final_nc, target_nc, 5, padding=2)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, n_up, crop_size):
        if n_up == 'normal':
            num_up_layers = 5
        elif n_up == 'more':
            num_up_layers = 6
        elif n_up == 'most':
            num_up_layers = 7

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             n_up)

        sw = crop_size // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.has_z:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                print("Missing z vector, sampling from normal")
                z = torch.randn(input.size(0), self.nz,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 12 * self.nf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        seg_1 = F.interpolate(seg, size=[self.sh, self.sw])
        x = self.head_0(x, seg_1)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.n_up == 'more' or \
           self.n_up == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.n_up == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img_pre(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x

class SPADEResnetBlock2(nn.Module):
    def __init__(self, fin, fout, norm, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.semantic_nc = semantic_nc
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm.replace('spectral', '')
        self.norm_0 = SPADE2(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE2(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE2(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class SPADE2(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            raise ValueError
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_preshared_depth = nn.Sequential(nn.Conv2d(1, nhidden//8, kernel_size=ks, padding=pw))
        self.mlp_preshared_label = nn.Sequential(nn.Conv2d(label_nc-1, nhidden//2, kernel_size=1, padding=0))
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(nhidden//8+nhidden//2, nhidden, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        preactv_depth = self.mlp_preshared_depth(segmap[:,0:1,:,:])
        preactv_label = self.mlp_preshared_label(segmap[:,1:,:,:])
        postactv_segmap = torch.cat((preactv_depth, preactv_label), dim=1)
        actv = self.mlp_shared(postactv_segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.acti = nn.LeakyReLU(0.2, True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.acti(bottle)

class ConvEncoder_PSP_SE(nn.Module):
    """ More powerful network as it seems simply increasing nz does not help """
    """ Try adding a SE and a PSP to model channel/spatial interactions???"""

    def __init__(self, input_nc, output_nc, nef, vae):
        super().__init__()
        # ndf = opt.ngf
        self.vae = vae
        self.layer1 = SEResBlock3(input_nc, nef, 1)
        self.layer2 = SEResBlock3(nef, nef*2, 2)
        self.layer3 = SEResBlock3(nef*2, nef * 4, 2)
        self.psp = PSPModule(nef * 4, nef * 8)
        self.layer4 = SEResBlock3(nef * 8, nef * 8, 2)
        self.layer5 = SEResBlock3(nef * 8, nef * 16, 2)

        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.actvn = nn.LeakyReLU(0.2, True)

        self.fc_mu = nn.Linear(nef * 16, output_nc)
        self.fc_var = nn.Linear(nef * 16, output_nc)
        self.fc_z = nn.Linear(nef * 16, output_nc)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.psp(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool_layer(x)
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        if self.vae:
            mu = self.fc_mu(x)
            logvar = self.fc_var(x)
            return mu, logvar
        else:
            z = self.fc_z(x)
            return z

class ConvEncoder_PSP_SE_MMD(nn.Module):
    """ More powerful network as it seems simply increasing nz does not help """
    """ Try adding a SE and a PSP to model channel/spatial interactions???"""

    def __init__(self, input_nc, output_nc, nef):
        super().__init__()
        # ndf = opt.ngf
        self.layer1 = SEResBlock3(input_nc, nef, 1)
        self.layer2 = SEResBlock3(nef, nef*2, 2)
        self.layer3 = SEResBlock3(nef*2, nef * 4, 2)
        self.psp = PSPModule(nef * 4, nef * 8)
        self.layer4 = SEResBlock3(nef * 8, nef * 8, 2)
        self.layer5 = SEResBlock3(nef * 8, nef * 16, 2)

        self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.actvn = nn.LeakyReLU(0.2, True)

        # self.fc_mu_pre = nn.Sequential(nn.Linear(nef * 16, 512), nn.ReLU(inplace=True))
        # self.fc_mu = nn.Linear(512, output_nc)
        #
        # self.fc_var_pre = nn.Sequential(nn.Linear(nef * 16, 512), nn.ReLU(inplace=True))
        # self.fc_var = nn.Linear(512, output_nc)

        self.fc_z_pre = nn.Sequential(nn.Linear(nef * 16, 512), nn.ReLU(inplace=True))
        self.fc_z = nn.Linear(512, output_nc)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.psp(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool_layer(x)
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        return self.fc_z(self.fc_z_pre(x))#, self.fc_mu(self.fc_mu_pre(x)),self.fc_var(self.fc_var_pre(x))

class ConvEncoder_PSP_SE_MMD_2(nn.Module):
    """ More powerful network as it seems simply increasing nz does not help """
    """ Try adding a SE and a PSP to model channel/spatial interactions???"""

    def __init__(self, input_nc, output_nc, nef):
        super().__init__()
        # ndf = opt.ngf
        self.layer1 = SEResBlock3(input_nc, nef, 2)
        self.layer2 = SEResBlock3(nef, nef*2, 2)
        self.layer3 = SEResBlock3(nef*2, nef * 4, 2)
        self.layer4 = SEResBlock3(nef * 4, nef * 8, 2)
        self.layer5 = SEResBlock3(nef * 8, nef * 16, 2)
        self.layer6 = SEResBlock3(nef * 16, nef * 16, 2)
        self.actvn = nn.LeakyReLU(0.2, True)
        self.fc_z_pre = nn.Sequential(nn.Linear(nef * 16 * 4 * 4, 512), nn.LeakyReLU(0.2, inplace=True))
        self.fc_z = nn.Linear(512, output_nc)

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        return self.fc_z(self.fc_z_pre(x))

class SPADE3(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            raise ValueError
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_preshared_depth = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(1, nhidden//8, kernel_size=ks, padding=0),nn.LeakyReLU(inplace=True))
        self.mlp_preshared_label = nn.Sequential(nn.Conv2d(label_nc-1, nhidden//2, kernel_size=1, padding=0),nn.LeakyReLU(inplace=True))
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(nhidden//8+nhidden//2, nhidden, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_gamma = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))
        self.mlp_beta = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        preactv_depth = self.mlp_preshared_depth(segmap[:,0:1,:,:])
        preactv_label = self.mlp_preshared_label(segmap[:,1:,:,:])
        postactv_segmap = torch.cat((preactv_depth, preactv_label), dim=1)
        actv = self.mlp_shared(postactv_segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class SPADEResnetBlock3(nn.Module):
    def __init__(self, fin, fout, norm, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.semantic_nc = semantic_nc
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        self.se = SEBlock2(fout, reduction=8)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm:
            self.conv_0 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_0))
            self.conv_1 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_1))
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm.replace('spectral', '')
        self.norm_0 = SPADE3(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE3(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE3(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        dx = self.se(dx)
        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class SPADEGenerator3(nn.Module):
    def __init__(self, semantic_nc, target_nc, nz, ngf, norm, crop_size, n_up):
        super().__init__()
        # self.opt = opt
        nf = ngf
        self.nf = ngf
        self.n_up = n_up
        self.sw, self.sh = self.compute_latent_vector_size(n_up, crop_size)
        self.has_z = nz>0
        self.nz = nz
        # todo: replace 8 with 16
        if self.has_z:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.nz, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock3(16 * nf, 16 * nf, norm, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock3(16 * nf, 16 * nf, norm, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock3(16 * nf, 16 * nf, norm, semantic_nc)

        self.up_0 = SPADEResnetBlock3(16 * nf, 8 * nf, norm, semantic_nc)
        self.up_1 = SPADEResnetBlock3(8 * nf, 4 * nf, norm, semantic_nc)
        self.up_2 = SPADEResnetBlock3(4 * nf, 2 * nf, norm, semantic_nc)
        self.up_3 = SPADEResnetBlock3(2 * nf, 1 * nf, norm, semantic_nc)

        final_nc = nf

        if n_up == 'most':
            self.up_4 = SPADEResnetBlock3(1 * nf, nf // 2, norm,semantic_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, target_nc, 5, padding=2)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, n_up, crop_size):
        if n_up == 'normal':
            num_up_layers = 5
        elif n_up == 'more':
            num_up_layers = 6
        elif n_up == 'most':
            num_up_layers = 7

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             n_up)

        sw = crop_size // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.has_z:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                print("Missing z vector, sampling from normal")
                z = torch.randn(input.size(0), self.nz,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.nf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        seg_1 = F.interpolate(seg, size=[self.sh, self.sw])
        x = self.head_0(x, seg_1)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.n_up == 'more' or \
           self.n_up == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.n_up == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x

class MultiscaleDiscriminator_MMD(nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--netD_subarch', type=str, default='n_layer',
    #                         help='architecture of each discriminator')
    #     parser.add_argument('--num_D', type=int, default=2,
    #                         help='number of discriminators to be used in multiscale')
    #     opt, _ = parser.parse_known_args()
    #
    #     # define properties of each discriminator of the multiscale discriminator
    #     subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
    #                                         'models.networks.discriminator')
    #     subnetD.modify_commandline_options(parser, is_train)
    #
    #     return parser

    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, num_D=2, use_feat_loss=True):
        super().__init__()
        self.use_feat_loss = use_feat_loss
        for i in range(num_D):
            subnetD = self.create_single_discriminator(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
            self.add_module('discriminator_%d' % i, subnetD)
            n_layers = n_layers - 1

    def create_single_discriminator(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            print("Selected n_layer pix2pixHD discrim")
            netD = NLayerDiscriminator_MMD(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.use_feat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# From SPADE
class NLayerDiscriminator_MMD(nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--n_layers_D', type=int, default=3,
    #                         help='# layers in each discriminator')
    #     return parser

    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss, nz=256):
        super().__init__()
        # self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf
        if conditional_nc <= 0:
            print("Creating Pix2PixHD discriminator")
            print("0 dimensional input set")
        input_nc_total = input_nc + conditional_nc

        norm_layer = get_nonspade_norm_layer(norm_layer)
        sequence = [[nn.Conv2d(input_nc_total, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]
        self.use_feat_loss = use_feat_loss
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride_val = 1 if n == n_layers - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride_val, padding=padw)),
                          nn.LeakyReLU(0.2, True)
                          ]]

        # sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        # sequence += [[nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=1)]]
        self.decide = nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=0)
        self.z_out = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),nn.LeakyReLU(inplace=True),nn.Conv2d(nf, nz, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d(1))
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    # def compute_D_input_nc(self, opt):
    #     input_nc = opt.label_nc + opt.output_nc
    #     if opt.contain_dontcare_label:
    #         input_nc += 1
    #     if not opt.no_instance:
    #         input_nc += 1
    #     return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.named_children():
            if ("decide" not in submodel[0]) and ("z_out" not in submodel[0]):
                intermediate_output = submodel[1](results[-1])
                results.append(intermediate_output)
        results.append((self.decide(results[-1]), self.z_out(results[-1])))
        get_intermediate_features = self.use_feat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]



class MultiscaleDiscriminator_MMD_2(nn.Module):


    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, num_D=2, use_feat_loss=True):
        super().__init__()
        self.use_feat_loss = use_feat_loss
        for i in range(num_D):
            subnetD = self.create_single_discriminator(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
            self.add_module('discriminator_%d' % i, subnetD)
            n_layers = n_layers - 1

    def create_single_discriminator(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            print("Selected n_layer pix2pixHD discrim")
            netD = NLayerDiscriminator_MMD_2(input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = self.use_feat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result

# From SPADE
class NLayerDiscriminator_MMD_2(nn.Module):
    # @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     parser.add_argument('--n_layers_D', type=int, default=3,
    #                         help='# layers in each discriminator')
    #     return parser

    def __init__(self, input_nc, conditional_nc, ndf, norm_layer, n_layers, use_feat_loss, nz=256):
        super().__init__()
        # self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = ndf
        if conditional_nc <= 0:
            print("Creating Pix2PixHD discriminator")
            print("0 dimensional input set")
        input_nc_total = input_nc + conditional_nc

        norm_layer = get_nonspade_norm_layer(norm_layer)
        sequence = [[nn.Conv2d(input_nc_total, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]
        self.use_feat_loss = use_feat_loss
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride_val = 1 if n == n_layers - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,stride=stride_val, padding=padw)),
                          nn.LeakyReLU(0.2, True)
                          ]]
        # sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        # sequence += [[nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=1)]]
        self.decide = nn.Conv2d(nf, 1, kernel_size=1, stride=1, padding=0)
        self.z_out = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(nf, nz, kernel_size=1, stride=1, padding=0), nn.AdaptiveAvgPool2d(1))
        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    # def compute_D_input_nc(self, opt):
    #     input_nc = opt.label_nc + opt.output_nc
    #     if opt.contain_dontcare_label:
    #         input_nc += 1
    #     if not opt.no_instance:
    #         input_nc += 1
    #     return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.named_children():
            if ("decide" not in submodel[0]) and ("z_out" not in submodel[0]):
                intermediate_output = submodel[1](results[-1])
                results.append(intermediate_output)
        results.append((self.decide(results[-1]), self.z_out(results[-1])))
        get_intermediate_features = self.use_feat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]





class SPADE4(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            raise ValueError
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'layer':
            self.param_free_norm = LayerNorm2D(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_preshared_depth = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(1, nhidden//8, kernel_size=ks, padding=0),nn.LeakyReLU(inplace=True))
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(nhidden//8+label_nc-1, nhidden, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))
        self.mlp_beta = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        preactv_depth = self.mlp_preshared_depth(segmap[:,0:1,:,:])
        postactv_segmap = torch.cat((preactv_depth, segmap[:,1:,:,:]), dim=1)
        actv = self.mlp_shared(postactv_segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock4(nn.Module):
    def __init__(self, fin, fout, norm, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.semantic_nc = semantic_nc
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        self.se = SEBlock2(fout, reduction=8)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm:
            self.conv_0 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_0))
            self.conv_1 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_1))
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm.replace('spectral', '')
        self.norm_0 = SPADE4(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE4(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE4(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        dx = self.se(dx)
        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class SPADEGenerator4(nn.Module):
    def __init__(self, semantic_nc, target_nc, nz, ngf, norm, crop_size, n_up):
        super().__init__()
        # self.opt = opt
        nf = ngf
        self.nf = ngf
        self.n_up = n_up
        self.sw, self.sh = self.compute_latent_vector_size(n_up, crop_size)
        self.has_z = nz>0
        self.nz = nz
        # todo: replace 8 with 16
        if self.has_z:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.nz, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock4(16 * nf, 16 * nf, norm, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock4(16 * nf, 16 * nf, norm, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock4(16 * nf, 16 * nf, norm, semantic_nc)

        self.up_0 = SPADEResnetBlock4(16 * nf, 8 * nf, norm, semantic_nc)
        self.up_1 = SPADEResnetBlock4(8 * nf, 4 * nf, norm, semantic_nc)
        self.up_2 = SPADEResnetBlock4(4 * nf, 2 * nf, norm, semantic_nc)
        self.up_3 = SPADEResnetBlock4(2 * nf, 1 * nf, norm, semantic_nc)

        final_nc = nf

        if n_up == 'most':
            self.up_4 = SPADEResnetBlock4(1 * nf, nf // 2, norm,semantic_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, target_nc, 5, padding=2)

        self.up_b = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_n = nn.Upsample(scale_factor=2, mode='nearest')
    def compute_latent_vector_size(self, n_up, crop_size):
        if n_up == 'normal':
            num_up_layers = 5
        elif n_up == 'more':
            num_up_layers = 6
        elif n_up == 'most':
            num_up_layers = 7

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             n_up)

        sw = crop_size // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.has_z:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                print("Missing z vector, sampling from normal")
                z = torch.randn(input.size(0), self.nz,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.nf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        seg_1 = F.interpolate(seg, size=[self.sh, self.sw])
        x = self.head_0(x, seg_1)

        x = self.up_n(x)
        x = self.G_middle_0(x, seg)

        if self.n_up == 'more' or \
           self.n_up == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        x = self.up_n(x)
        x = self.up_0(x, seg)
        x = self.up_n(x)
        x = self.up_1(x, seg)
        x = self.up_n(x)
        x = self.up_2(x, seg)
        x = self.up_b(x)
        x = self.up_3(x, seg)

        if self.n_up == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x

class SPADE5(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            raise ValueError
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'layer':
            self.param_free_norm = LayerNorm2D(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_preshared_depth = nn.Sequential(nn.ReflectionPad2d(pw),
                                                 nn.Conv2d(1, 40, kernel_size=ks, padding=0),
                                                 nn.Tanh())
        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(80, nhidden, kernel_size=3, padding=0),
            nn.LeakyReLU(inplace=True)
        )
        self.mlp_gamma = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))
        self.mlp_beta = nn.Sequential(nn.ReflectionPad2d(pw), nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0))

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        preactv_depth = self.mlp_preshared_depth(segmap[:, 0:1, :, :])*segmap[:,1:,:,:]
        postactv_segmap = torch.cat((preactv_depth, segmap[:,1:,:,:]), dim=1)
        actv = self.mlp_shared(postactv_segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
class SPADEResnetBlock5(nn.Module):
    def __init__(self, fin, fout, norm, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        self.semantic_nc = semantic_nc
        fmiddle = min(fin, fout)
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in norm:
            self.conv_0 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_0))
            self.conv_1 = nn.Sequential(nn.ReflectionPad2d(1),spectral_norm(self.conv_1))
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = norm.replace('spectral', '')
        self.norm_0 = SPADE5(spade_config_str, fin, self.semantic_nc)
        self.norm_1 = SPADE5(spade_config_str, fmiddle, self.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE5(spade_config_str, fin, self.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)

class SPADEGenerator5(nn.Module):
    def __init__(self, semantic_nc, target_nc, nz, ngf, norm, crop_size, n_up):
        super().__init__()
        # self.opt = opt
        nf = ngf
        self.nf = ngf
        self.n_up = n_up
        self.sw, self.sh = self.compute_latent_vector_size(n_up, crop_size)
        self.has_z = nz>0
        self.nz = nz
        # todo: replace 8 with 16
        if self.has_z:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(self.nz, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock5(16 * nf, 16 * nf, norm, semantic_nc)

        self.G_middle_0 = SPADEResnetBlock5(16 * nf, 16 * nf, norm, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock5(16 * nf, 16 * nf, norm, semantic_nc)

        self.up_0 = SPADEResnetBlock5(16 * nf, 8 * nf, norm, semantic_nc)
        self.up_1 = SPADEResnetBlock5(8 * nf, 4 * nf, norm, semantic_nc)
        self.up_2 = SPADEResnetBlock5(4 * nf, 2 * nf, norm, semantic_nc)
        self.up_3 = SPADEResnetBlock5(2 * nf, 1 * nf, norm, semantic_nc)

        final_nc = nf

        if n_up == 'most':
            self.up_4 = SPADEResnetBlock4(1 * nf, nf // 2, norm,semantic_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, target_nc, 3, padding=1)

        self.up_b = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_n = nn.Upsample(scale_factor=2, mode='nearest')
    def compute_latent_vector_size(self, n_up, crop_size):
        if n_up == 'normal':
            num_up_layers = 5
        elif n_up == 'more':
            num_up_layers = 6
        elif n_up == 'most':
            num_up_layers = 7

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             n_up)

        sw = crop_size // (2**num_up_layers)
        sh = sw

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.has_z:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                print("Missing z vector, sampling from normal")
                z = torch.randn(input.size(0), self.nz,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.nf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        seg_1 = F.interpolate(seg, size=[self.sh, self.sw])
        x = self.head_0(x, seg_1)

        x = self.up_n(x)
        x = self.G_middle_0(x, seg)

        if self.n_up == 'more' or \
           self.n_up == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)
        x = self.up_n(x)
        x = self.up_0(x, seg)
        x = self.up_n(x)
        x = self.up_1(x, seg)
        x = self.up_n(x)
        x = self.up_2(x, seg)
        x = self.up_b(x)
        x = self.up_3(x, seg)

        if self.n_up == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1, inplace=True))
        x = F.tanh(x)

        return x