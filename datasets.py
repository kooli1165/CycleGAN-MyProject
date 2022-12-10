import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class DingziLabelDataset(Dataset):
    def __init__(self, img_path, attr_path, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        self.files_A = sorted(glob.glob(os.path.join(img_path, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(img_path, '%sB' % mode) + '/*.*'))
        self.attr_path = attr_path
        self.attr2idx = {}
        self.idx2attr = {}
        self.filename2label = {}
        self.preprocess()

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in all_attr_names:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            self.filename2label[filename] = label

    def __getitem__(self, index):
        i_a = index % len(self.files_A)

        item_A = self.transform(Image.open(self.files_A[i_a]))
        filename_A = os.path.basename(self.files_A[i_a])
        label_A = self.filename2label[filename_A]

        if self.unaligned:
            i_b = random.randint(0, len(self.files_B) - 1)
        else:
            i_b = index % len(self.files_B)

        item_B = self.transform(Image.open(self.files_B[i_b]))
        filename_B = os.path.basename(self.files_B[i_b])
        label_B = self.filename2label[filename_B]

        return {'A': [item_A, label_A], 'B': [item_B, label_B]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

