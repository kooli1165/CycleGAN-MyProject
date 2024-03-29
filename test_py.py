#!/usr/bin/python3

import argparse
import sys
import os
from datetime import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import MyGenerator_v0_1
from models import MyGenerator_v0_1_ximg
from models import MyGenerator_v0_3
from models import MyGenerator_v0_3_ximg
from models import Generator
from datasets import ImageDataset
from datasets import ImageDatasetWithFilename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='v4.0_dataset1.0_all_ximg')
    parser.add_argument('--eval_mode', type=str, default='train')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./datasets/dingzi_v1_0_all/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/v4.0_dataset1.0/best_netG_A2B.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/v4.0_dataset1.0/best_netG_B2A.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    with_mask = True

    ###### Definition of variables ######
    # Networks
    netG_A2B = MyGenerator_v0_3_ximg(opt.input_nc, opt.output_nc)
    netG_B2A = MyGenerator_v0_3_ximg(opt.output_nc, opt.input_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDatasetWithFilename(opt.dataroot, transforms_=transforms_, mode=opt.eval_mode),
                            batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    output_dirname = ''
    if not opt.save_name:
        output_dirname = datetime.now().strftime('%b%d_%H-%M-%S')
    else:
        output_dirname = opt.save_name

    save_dir = os.path.join('results', output_dirname)

    if not os.path.exists(save_dir + '/fake_A'):
        os.makedirs(save_dir + '/fake_A')
    if not os.path.exists(save_dir + '/fake_B'):
        os.makedirs(save_dir + '/fake_B')

    if with_mask:
        if not os.path.exists(save_dir + '/fake_A_mask'):
            os.makedirs(save_dir + '/fake_A_mask')
        if not os.path.exists(save_dir + '/fake_B_mask'):
            os.makedirs(save_dir + '/fake_B_mask')
        if not os.path.exists(save_dir + '/fake_A_img'):
            os.makedirs(save_dir + '/fake_A_img')
        if not os.path.exists(save_dir + '/fake_B_img'):
            os.makedirs(save_dir + '/fake_B_img')

    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generate output
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)

        if with_mask:
            fake_B_img = fake_B[2]
            fake_A_img = fake_A[2]
            fake_B_img = 0.5 * (fake_B_img.data + 1.0)
            fake_A_img = 0.5 * (fake_A_img.data + 1.0)
            fake_B_mask = fake_B[1]
            fake_A_mask = fake_A[1]
            fake_B = fake_B[0]
            fake_A = fake_A[0]

        fake_B = 0.5 * (fake_B.data + 1.0)
        fake_A = 0.5 * (fake_A.data + 1.0)

        # Save image files
        B_name = os.path.basename(batch['A_filename'][0])
        A_name = os.path.basename(batch['B_filename'][0])
        save_image(fake_A, os.path.join(save_dir, 'fake_A', A_name))
        save_image(fake_B, os.path.join(save_dir, 'fake_B', B_name))

        if with_mask:
            save_image(fake_A_mask.data, os.path.join(save_dir, 'fake_A_mask', A_name))
            save_image(fake_B_mask.data, os.path.join(save_dir, 'fake_B_mask', B_name))

            save_image(fake_A_img.data, os.path.join(save_dir, 'fake_A_img', A_name))
            save_image(fake_B_img.data, os.path.join(save_dir, 'fake_B_img', B_name))

            b_name, b_extension = os.path.splitext(B_name)
            a_name, b_extension = os.path.splitext(A_name)

            fake_B_m_r = fake_B_mask[:, 0, :, :]
            fake_B_m_g = fake_B_mask[:, 1, :, :]
            fake_B_m_b = fake_B_mask[:, 2, :, :]

            save_image(fake_B_m_r.data, os.path.join(save_dir, 'fake_B_mask', b_name + '_r.png'))
            save_image(fake_B_m_g.data, os.path.join(save_dir, 'fake_B_mask', b_name + '_g.png'))
            save_image(fake_B_m_b.data, os.path.join(save_dir, 'fake_B_mask', b_name + '_b.png'))

            fake_A_m_r = fake_A_mask[:, 0, :, :]
            fake_A_m_g = fake_A_mask[:, 1, :, :]
            fake_A_m_b = fake_A_mask[:, 2, :, :]

            save_image(fake_A_m_r.data, os.path.join(save_dir, 'fake_A_mask', a_name + '_r.png'))
            save_image(fake_A_m_g.data, os.path.join(save_dir, 'fake_A_mask', a_name + '_g.png'))
            save_image(fake_A_m_b.data, os.path.join(save_dir, 'fake_A_mask', a_name + '_b.png'))


        sys.stdout.write('\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

    sys.stdout.write('\n')
    ###################################
