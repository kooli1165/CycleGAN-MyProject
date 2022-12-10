#!/usr/bin/python3

import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import MyGenerator
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./output/v0.3_dataset1.0')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/dingzi_v1_0/',
                        help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--lr_D', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_every_n_epoch', type=int, default=50)
    parser.add_argument('--n_train_D', type=int, default=5, help='每多少次迭代训练一次D')
    parser.add_argument('--lambda_G', type=float, default=1.0)
    parser.add_argument('--lambda_Idt', type=float, default=5.0)
    parser.add_argument('--lambda_Cycle', type=float, default=5.0)
    parser.add_argument('--lambda_D', type=float, default=0.5)
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = MyGenerator(opt.input_nc, opt.output_nc)
    netG_B2A = MyGenerator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    # criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    # input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    # input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    losses = {}
    # loss_D_A = 0.0
    # loss_D_B = 0.0
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            input_A = Tensor(batch['A'].shape[0], opt.input_nc, opt.size, opt.size)
            input_B = Tensor(batch['B'].shape[0], opt.output_nc, opt.size, opt.size)
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B, same_B_mask = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * opt.lambda_Idt
            # G_B2A(A) should equal A if real A is fed
            same_A, same_A_mask = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * opt.lambda_Idt

            # GAN loss
            fake_B, fake_B_mask = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * opt.lambda_G

            fake_A, fake_A_mask = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * opt.lambda_G

            # Cycle loss
            recovered_A, recovered_A_mask = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.lambda_Cycle

            recovered_B, recovered_B_mask = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * opt.lambda_Cycle

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            if (i+1) % opt.n_train_D == 0:
                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * opt.lambda_D
                loss_D_A.backward()

                optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * opt.lambda_D
                loss_D_B.backward()

                optimizer_D_B.step()
                ###################################
                losses['loss_D'] = (loss_D_A + loss_D_B)

            # Progress report (http://localhost:8097)
            losses['loss_G'] = loss_G
            losses['loss_G_identity'] =  (loss_identity_A + loss_identity_B)
            losses['loss_G_GAN'] = (loss_GAN_A2B + loss_GAN_B2A)
            losses['loss_G_cycle'] = (loss_cycle_ABA + loss_cycle_BAB)
            losses['loss_G'] = loss_G

            logger.log(losses,
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B,
                               'fake_A_mask':fake_A_mask, 'fake_B_mask':fake_B_mask})

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        torch.save(netG_A2B.state_dict(), opt.save_dir + '/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), opt.save_dir + '/netG_B2A.pth')
        torch.save(netD_A.state_dict(), opt.save_dir + '/netD_A.pth')
        torch.save(netD_B.state_dict(), opt.save_dir + '/netD_B.pth')

        if epoch % opt.save_every_n_epoch == 0:
            torch.save(netG_A2B.state_dict(), opt.save_dir + '/%d_netG_A2B.pth' % epoch)
            torch.save(netG_B2A.state_dict(), opt.save_dir + '/%d_netG_B2A.pth' % epoch)
            torch.save(netD_A.state_dict(), opt.save_dir + '/%d_netD_A.pth' % epoch)
            torch.save(netD_B.state_dict(), opt.save_dir + '/%d_netD_B.pth' % epoch)
    ###################################

    flog = open(os.path.join(opt.save_dir, 'log_opt'), 'w')
    logs = opt + '\n'
    flog.write(logs)
    flog.close()
