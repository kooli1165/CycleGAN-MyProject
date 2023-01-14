#!/usr/bin/python3

import argparse
import itertools
import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import MyGenerator_v0_1
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import MyLambdaLR
from utils import Logger
from utils import TensorboardLogger
from utils import weights_init_normal
from utils import run_netD
from utils import d_logistic_loss
from utils import d_r1_loss
from utils import g_nonsaturating_loss
from utils import g_path_regularize
from utils import compute_gradient_penalty
from lib_add.augment_ori import AugmentPipe
from datasets import ImageDataset

from metrics.eval import eval_metrics

# v1.1  修改损失函数以适配 自适应p
# v1.2  (调整 G D 训练策略  每n次迭代训练一次G)
#          训练过程中自动测试fid
#          ada_kimg 80->8
#          lambda_D 0.5 -> 1
# v1.2.1  ada_kimg 8->0.6
#         ada_interval 4->16
#         每10epoch 测试fid
# v2.0   G 替换为3维mask的mygeneratorv0.1   ada_interval 16->4 ada_target 0.6 -> 0.5
#           fid 99.57 epoch 261
# v2.0.1   继承2.0 调整学习率0.0002->0.0001 decay_epoch
# v2.0.2   继承2.0.1  关闭自带的数据增强
# v2.0.3   继承2.0.1  关闭自带的数据增强  ada_interval 4->16
# v2.0.4   继承2.0.2  关闭自带的数据增强  调整学习率下降策略  v2.0.4.1  100.69 in 109
# v2.1.3   关闭ada

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./output/v2.1.3_dataset1.0')
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/dingzi_v1_0/',
                        help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--min_lr_D', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_every_n_epoch', type=int, default=200)
    parser.add_argument('--eval_every_n_epoch', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='test')

    parser.add_argument('--n_train_D', type=int, default=1, help='每多少次迭代训练一次D')
    parser.add_argument('--n_train_G', type=int, default=1, help='每多少次迭代训练一次G')
    parser.add_argument('--lambda_G', type=float, default=1.0)
    parser.add_argument('--lambda_Idt', type=float, default=5.0)
    parser.add_argument('--lambda_Cycle', type=float, default=5.0)
    parser.add_argument('--lambda_D', type=float, default=1.0)
    # ADA
    parser.add_argument('--ada_start_p', type=float, default=0)
    parser.add_argument('--ada_target', type=float, default=0.5)
    parser.add_argument('--ada_interval', type=int, default=4)
    parser.add_argument('--ada_kimg', type=float, default=0.6)
    parser.add_argument('--ada_fixed', type=bool, default=False)
    # gp loss
    parser.add_argument('--d_r1', type=bool, default=True)
    parser.add_argument('--r1_gamma', type=float, default=2.1845)  #3.2768    0.8192
    parser.add_argument('--d_r1_every_n', type=float, default=16)
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # ADA
    # augment_pipe = AugmentPipe(opt.ada_start_p, opt.ada_target, opt.ada_interval, opt.ada_kimg).train()
    augment_pipe = None

    # Networks
    netG_A2B = MyGenerator_v0_1(opt.input_nc, opt.output_nc)
    netG_B2A = MyGenerator_v0_1(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    with_mask = True

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()
        if augment_pipe is not None:
            augment_pipe.cuda()

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

    # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
    #                                                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
    #                                                                       opt.decay_epoch).step)
    # lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
    #                                                      lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
    #                                                                         opt.decay_epoch).step)
    # lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
    #                                                      lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
    #                                                                         opt.decay_epoch).step)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=MyLambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch, opt.lr, opt.min_lr).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=MyLambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch, opt.lr_D, opt.min_lr_D).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=MyLambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch, opt.lr_D, opt.min_lr_D).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    # input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    # input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [
                   # transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   # transforms.RandomCrop(opt.size),
                   # transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    transforms_eval = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader_eval = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_eval, mode=opt.eval_mode),
                                 batch_size=1, shuffle=False, num_workers=opt.n_cpu)

    # eval
    best_fid = sys.maxsize
    best_fid_epoch = 0

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    tb_logger = TensorboardLogger(output_path=opt.save_dir)
    losses = {}
    # loss_D_A = 0.0
    # loss_D_B = 0.0
    ###################################

    i_G = 0  # 训练G 的迭代次数
    i_D = 0  # 训练D 的迭代次数

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):

        for i, batch in enumerate(dataloader):
            ## Set model input
            input_A = Tensor(batch['A'].shape[0], opt.input_nc, opt.size, opt.size)
            input_B = Tensor(batch['B'].shape[0], opt.output_nc, opt.size, opt.size)
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            if (i + 1) % opt.n_train_G == 0:
                i_G = i_G + 1

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            ## Identity loss
            # G_A2B(B) should equal B if real B is fed
            # same_B, same_B_mask = netG_A2B(real_B)
            same_B = netG_A2B(real_B)
            if with_mask: same_B = same_B[0]
            loss_identity_B = criterion_identity(same_B, real_B) * opt.lambda_Idt
            # G_B2A(A) should equal A if real A is fed
            # same_A, same_A_mask = netG_B2A(real_A)
            same_A = netG_B2A(real_A)
            if with_mask: same_A = same_A[0]
            loss_identity_A = criterion_identity(same_A, real_A) * opt.lambda_Idt

            ## GAN loss
            fake_B, fake_B_mask = netG_A2B(real_A)
            # fake_B = netG_A2B(real_A)
            # pred_fake = netD_B(fake_B)
            pred_fake = run_netD(fake_B, netD_B, augment_pipe)
            # loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * opt.lambda_G
            loss_GAN_A2B = g_nonsaturating_loss(pred_fake) * opt.lambda_G

            fake_A, fake_A_mask = netG_B2A(real_B)
            # fake_A = netG_B2A(real_B)
            # pred_fake = netD_A(fake_A)
            pred_fake = run_netD(fake_A, netD_A, augment_pipe)
            # loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * opt.lambda_G
            loss_GAN_B2A = g_nonsaturating_loss(pred_fake) * opt.lambda_G

            ## Cycle loss
            # recovered_A, recovered_A_mask = netG_B2A(fake_B)
            recovered_A = netG_B2A(fake_B)
            if with_mask: recovered_A = recovered_A[0]
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.lambda_Cycle

            ## recovered_B, recovered_B_mask = netG_A2B(fake_A)
            recovered_B = netG_A2B(fake_A)
            if with_mask: recovered_B = recovered_B[0]
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * opt.lambda_Cycle

            ## Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()

            ## G loss log
            losses['loss_G'] = loss_G
            losses['loss_G_identity'] =  (loss_identity_A + loss_identity_B)
            losses['loss_G_GAN'] = (loss_GAN_A2B + loss_GAN_B2A)
            losses['loss_G_cycle'] = (loss_cycle_ABA + loss_cycle_BAB)

            ###################################

            if (i+1) % opt.n_train_D == 0:
                i_D = i_D + 1

                ###### Discriminator A ######
                optimizer_D_A.zero_grad()

                ## Real loss
                # pred_real = netD_A(real_A)
                pred_real = run_netD(real_A, netD_A, augment_pipe)
                # loss_D_real = criterion_GAN(pred_real, target_real)

                if augment_pipe is not None:
                    augment_pipe.accumulate_real_sign(pred_real.sign().detach())

                ## Fake loss
                # with torch.no_grad():
                #     fake_A = netG_B2A(real_B)
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                # pred_fake = netD_A(fake_A.detach())
                pred_fake = run_netD(fake_A.detach(), netD_A, augment_pipe)
                # loss_D_fake = criterion_GAN(pred_fake, target_fake)

                ## Total loss
                # loss_D_A = (loss_D_real + loss_D_fake) * opt.lambda_D
                loss_D_A = d_logistic_loss(pred_real, pred_fake) * opt.lambda_D
                loss_D_A.backward()

                optimizer_D_A.step()

                ## d_r1 loss    gp
                if opt.d_r1 and ((i_D + 1) % opt.d_r1_every_n == 0):
                    real_A.requires_grad = True

                    real_pred = run_netD(real_A, netD_A, augment_pipe)
                    r1_loss = d_r1_loss(real_pred, real_A)

                    optimizer_D_A.zero_grad()
                    (opt.r1_gamma / 2 * r1_loss * opt.d_r1_every_n).backward()

                    optimizer_D_A.step()

                    losses['loss_D_A_r1'] = r1_loss

                ###################################

                ###### Discriminator B ######
                optimizer_D_B.zero_grad()

                ## Real loss
                # pred_real = netD_B(real_B)
                pred_real = run_netD(real_B, netD_B, augment_pipe)
                # loss_D_real = criterion_GAN(pred_real, target_real)

                if augment_pipe is not None:
                    augment_pipe.accumulate_real_sign(pred_real.sign().detach())

                ## Fake loss
                # with torch.no_grad():
                #     fake_B = netG_A2B(real_A)
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                # pred_fake = netD_B(fake_B.detach())
                pred_fake = run_netD(fake_B.detach(), netD_B, augment_pipe)
                # loss_D_fake = criterion_GAN(pred_fake, target_fake)

                ## Total loss
                # loss_D_B = (loss_D_real + loss_D_fake) * opt.lambda_D
                loss_D_B = d_logistic_loss(pred_real, pred_fake) * opt.lambda_D
                loss_D_B.backward()

                optimizer_D_B.step()

                ## d_r1 loss    gp
                if opt.d_r1 and ((i_D + 1) % opt.d_r1_every_n == 0):
                    real_B.requires_grad = True

                    real_pred = run_netD(real_B, netD_B, augment_pipe)
                    r1_loss = d_r1_loss(real_pred, real_B)

                    optimizer_D_B.zero_grad()
                    (opt.r1_gamma / 2 * r1_loss * opt.d_r1_every_n).backward()

                    optimizer_D_B.step()

                    losses['loss_D_B_r1'] = r1_loss

                ###################################
                losses['loss_D'] = (loss_D_A + loss_D_B)

                if augment_pipe is not None:
                    if (i_D+1) % opt.ada_interval == 0:
                        p_real_signs = augment_pipe.heuristic_update(opt.batchSize)
                        # losses['rt'] = p_real_signs
                        tb_logger.add_scalars({'rt': p_real_signs}, epoch * opt.batchSize + i)
                    losses['ada_p'] = augment_pipe.p

            ## Progress report (http://localhost:8097)

            logger.log(losses,
                       # images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B,
                               'fake_A_mask':fake_A_mask, 'fake_B_mask':fake_B_mask})

            tb_logger.add_scalars(losses, epoch * opt.batchSize + i)

        ## Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        ## Save models checkpoints
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        torch.save(netG_A2B.state_dict(), opt.save_dir + '/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), opt.save_dir + '/netG_B2A.pth')
        torch.save(netD_A.state_dict(), opt.save_dir + '/netD_A.pth')
        torch.save(netD_B.state_dict(), opt.save_dir + '/netD_B.pth')

        # eval model metrics
        if (epoch+1) % opt.eval_every_n_epoch == 0:
            fid = eval_metrics(netG_A2B, dataloader_eval, os.path.join(opt.dataroot, '%sB' % opt.eval_mode), with_mask=with_mask)
            tb_logger.add_scalars({'fid': fid}, epoch+1)

            if fid < best_fid:
                best_fid = fid
                best_fid_epoch = epoch + 1

                torch.save(netG_A2B.state_dict(), opt.save_dir + '/best_netG_A2B.pth')
                torch.save(netG_B2A.state_dict(), opt.save_dir + '/best_netG_B2A.pth')
                torch.save(netD_A.state_dict(), opt.save_dir + '/best_netD_A.pth')
                torch.save(netD_B.state_dict(), opt.save_dir + '/best_netD_B.pth')

        if epoch % opt.save_every_n_epoch == 0:
            torch.save(netG_A2B.state_dict(), opt.save_dir + '/%d_netG_A2B.pth' % epoch)
            torch.save(netG_B2A.state_dict(), opt.save_dir + '/%d_netG_B2A.pth' % epoch)
            torch.save(netD_A.state_dict(), opt.save_dir + '/%d_netD_A.pth' % epoch)
            torch.save(netD_B.state_dict(), opt.save_dir + '/%d_netD_B.pth' % epoch)
    ###################################

    print('best_fid: {0}, in epoch {1}'.format(best_fid, best_fid_epoch))
    flog = open(os.path.join(opt.save_dir, 'log_opt.txt'), 'w')
    logs = vars(opt)
    flog.write(str(logs) + '\n')
    flog.write('best_fid: {0}, in epoch {1}'.format(best_fid, best_fid_epoch))
    flog.close()
