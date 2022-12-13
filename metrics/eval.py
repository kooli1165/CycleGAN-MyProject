import os
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from pytorch_fid import fid_score


def eval_fid(paths):

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)

    fid_value = fid_score.calculate_fid_given_paths(paths=paths,
                                          batch_size=50,
                                          device=device,
                                          dims=2048,
                                          num_workers=num_workers)
    print('FID: ', fid_value)
    return fid_value


def eval_metrics(netG_A2B, dataloader, real_path, fake_path=None, metric='fid', with_mask=False):
    if fake_path is None:
        fake_path = os.path.join('./eval_tmp')
    if not os.path.exists(fake_path):
        os.makedirs(fake_path)

    Tensor = torch.cuda.FloatTensor if (torch.cuda.is_available()) else torch.Tensor
    for i, batch in enumerate(dataloader):
        input_A = Tensor(batch['A'].shape[0], batch['A'].shape[1], batch['A'].shape[2], batch['A'].shape[3])
        real_A = Variable(input_A.copy_(batch['A']))

        # Generate output
        with torch.no_grad():
            fake_B = netG_A2B(real_A)
            if with_mask:
                fake_B = fake_B[0]
            # fake_B, fake_B_mask = netG_A2B(real_A)
            # fake_A, fake_A_mask = netG_B2A(real_B)
            fake_B = 0.5 * (fake_B.data + 1.0)

        save_image(fake_B, os.path.join(fake_path, '%04d.png' % (i + 1)))

    value = 0
    if metric == 'fid':
        value = eval_fid([real_path, fake_path])

    shutil.rmtree(fake_path)

    return value


