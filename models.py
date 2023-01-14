import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class MyGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        self.model = nn.Sequential(*model)

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, 1, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = mask * img + (1 - mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask


class MyGenerator_v0_1(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator_v0_1, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        self.model = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model_img += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model_mask += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = mask * img + (1 - mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask


class MyGenerator_v0_1_ximg(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator_v0_1_ximg, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        self.model = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model_img += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model_mask += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = mask * img + (1 - mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask, img

class MyGenerator_v0_2(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator_v0_2, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        self.model = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model_img += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model_mask += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = (1 - mask) * img + (mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask


# pixelShuffle
class MyGenerator_v0_3(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator_v0_3, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        self.model = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model_img += [
                        nn.Conv2d(in_features, out_features*4, 1),
                        nn.PixelShuffle(2),
                        # nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model_mask += [
                          # nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.Conv2d(in_features, out_features * 4, 1),
                          nn.PixelShuffle(2),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = mask * img + (1 - mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask



class MyGenerator_v0_3_ximg(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(MyGenerator_v0_3_ximg, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for i in range(n_residual_blocks):
            res_blk = ResidualBlock(in_features)
            res_blk.weight_init(0, 0.2)
            model += [res_blk]

        self.model = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model_img = []
        model_mask = []
        for _ in range(2):
            model_img += [
                        nn.Conv2d(in_features, out_features*4, 1),
                        nn.PixelShuffle(2),
                        # nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            model_mask += [
                          # nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                          nn.Conv2d(in_features, out_features * 4, 1),
                          nn.PixelShuffle(2),
                          nn.InstanceNorm2d(out_features),
                          nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model_img += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        model_mask += [  nn.ReflectionPad2d(3),
                    # nn.Conv2d(64, output_nc, 7),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        # self.rgb_tanh = nn.Tanh()
        # self.a_sigmod = nn.Sigmoid()

        self.model_img = nn.Sequential(*model_img)
        self.model_mask = nn.Sequential(*model_mask)

    def forward(self, x):
        ori_img = x

        x = self.model(x)

        img = self.model_img(x)
        mask = self.model_mask(x)

        # mask_img = mask[:, 0:3, :, :]
        # mask_ori = mask[:, 3, :, :].unsqueeze(1)

        x = mask * img + (1 - mask) * ori_img

        # a = x[:, 3, :, :].unsqueeze(1)
        # a = 0.5 * (a + 1)
        #
        # fake_img = x[:, 0:3, :, :]
        #
        # result = a * fake_img + (1 - a) * ori_img
        return x, mask, img



class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class MyDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(MyDiscriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class StarganDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=256, conv_dim=64, c_dim=4, repeat_num=6):
        super(StarganDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        # out_cls = self.conv2(h)
        # return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src
