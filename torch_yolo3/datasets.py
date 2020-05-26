import glob
import os
import random
import logging
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from torch_yolo3.augment import horisontal_flip, vertical_flip
from torch_yolo3.utils import update_path


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    """Infinite loading images from folder.

    >>> im_dir = ImageFolder(update_path(os.path.join('data', 'samples')))
    >>> len(im_dir)
    9
    >>> p_im, img = im_dir[0]
    >>> p_im  # doctest: +ELLIPSIS
    '...data/samples/dog.jpg'
    >>> img.shape
    torch.Size([3, 416, 416])
    """
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif')

    def __init__(self, folder_path, img_size=416):
        self.files = sorted([p for p in glob.glob(os.path.join(folder_path, "*.*"))
                             if os.path.splitext(os.path.basename(p))[1] in self.IMAGE_EXTENSIONS])
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        # in case of PNG image use only RGB not alpha
        return img_path, img[:3, ...]

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    """Loading data from a list

    >>> data = ListDataset(update_path(os.path.join('data', 'custom', 'train.txt')))
    >>> len(data)
    1
    >>> p_im, img, det = data[0]
    >>> p_im  # doctest: +ELLIPSIS
    '.../data/custom/images/coach.jpg'
    >>> img.shape
    torch.Size([3, 500, 500])
    >>> det.shape
    torch.Size([1, 6])
    >>> batch = [data[0], data[1]]
    >>> data.collate_fn(batch)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    (('.../images/coach.jpg', '.../images/coach.jpg'), tensor(...), tensor([[...], [...]]))
    """
    def __init__(self, list_path, img_size=416, augment=None, normalized_labels=True, max_objects=100):
        """Dataset defined by list of images/annot

        :param list list_path:
        :param int img_size:
        :param dict augment: options like {'scaling': 0.1, 'vflip': None, 'hflip': None}
        :param bool normalized_labels:
        :param int max_objects:
        """
        with open(list_path, "r") as file:
            img_files = [update_path(fn.strip()) for fn in file.readlines()]

        # get the labels for each image if image exista
        label_files = [
            os.path.join(os.path.dirname(imp).replace("images", "labels"),
                         os.path.splitext(os.path.basename(imp))[0] + ".txt")
            for imp in img_files if os.path.isfile(imp)
        ]
        logging.debug("From %i listed, found %i images", len(img_files), len(label_files))
        # filter existing image and annotation
        path_img_lbs = [(p_img, p_lbs) for p_img, p_lbs in zip(img_files, label_files)
                        if os.path.isfile(p_img) and os.path.isfile(p_lbs)]
        assert path_img_lbs, 'missing images/annotations'
        self.img_files, self.label_files = list(zip(*path_img_lbs))
        logging.debug("From %i listed found %i annotation", len(path_img_lbs), len(self.label_files))

        self.augment = augment if augment else {}
        self.img_size = img_size
        self.max_objects = max_objects
        self.normalized_labels = normalized_labels
        self.batch_count = 0

        scaling = self.augment.get('scaling', 0.)
        self.min_size = round(self.img_size * (1. - scaling) / 32) * 32
        self.max_size = round(self.img_size * (1. + scaling) / 32) * 32

    def __getitem__(self, idx):
        # ---------
        #  Image
        # ---------
        idx = idx % len(self.img_files)
        img_path = self.img_files[idx].rstrip()
        img_path = update_path(img_path)

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[idx].rstrip()
        label_path = update_path(label_path)

        targets = None
        if os.path.exists(label_path):
            with warnings.catch_warnings(record=True) as w:
                data = np.loadtxt(label_path)
            # in case the file is empty
            if len(data) > 0:
                boxes = torch.from_numpy(data.reshape(-1, 5))
                targets = ListDataset._convert_boxes_to_targets(
                    boxes, pad, (w_factor, h_factor), (padded_w, padded_h))

        # Apply augmentations
        if self.augment.get('hflip', False) and np.random.random() < 0.5:
            img, targets = horisontal_flip(img, targets)
        if self.augment.get('vflip', False) and np.random.random() < 0.5:
            img, targets = vertical_flip(img, targets)

        return img_path, img, targets

    @staticmethod
    def _convert_boxes_to_targets(boxes, pad, factor, padded):
        w_factor, h_factor = factor
        padded_w, padded_h = padded
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        return targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0) if len(targets) > 0 else torch.tensor(np.empty([0, 6]))
        # Selects new image size every tenth batch
        img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
