import os
import os.path as osp
import sys
import glob
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from PIL import Image


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


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target, height, width):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip() 
            bbox = obj.find('bndbox')
            if name.startswith('j') or name.startswith('w'):
                label = 0
            elif name.startswith('z') or name.startswith('p'):
                label = 1
            elif name.startswith('s') or name.startswith('i'):
                label = 2
            elif name.startswith('l'):
                label = 3
            elif name.startswith('d'):
                label = 4
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            bndbox.append(label)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ] 


class ListDataset(Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, image_path, info_path, img_size=1216, augment=True, multiscale=True, normalized_labels=True):
        self.image_path = image_path 
        self.info_path = info_path 
        self.target_transform = VOCAnnotationTransform()
        
        self.images = os.listdir(self.image_path) 
        self.ids = list() 

        if augment is True:
            fo = open("utils/train_aug.txt","r")
            for line in fo.readlines():
                line = line.strip()
                self.ids.append(line)
        else:
            fo = open("utils/val_aug.txt","r")
            for line in fo.readlines():
                line = line.strip()
                self.ids.append(line)

        self._annopath = osp.join(self.info_path, '%s.xml')
        self._imgpath = osp.join(self.image_path, '%s.jpg')
        self.normalized_labels = normalized_labels
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        img_id = self.ids[index]
        boxes = ET.parse(self._annopath % img_id).getroot()
        img_path = self._imgpath % img_id
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
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
        if self.target_transform is not None:
            boxes = self.target_transform(boxes,h,w) # [xmin, ymin, xmax, ymax, label_idx]; normalized; type:list
        
        boxes = torch.from_numpy(np.array(boxes).reshape(-1, 5))
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * boxes[:, 0]
        y1 = h_factor * boxes[:, 1]
        x2 = w_factor * boxes[:, 2]
        y2 = h_factor * boxes[:, 3]
    
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 0] = ((x1 + x2) / 2) / padded_w
        boxes[:, 1] = ((y1 + y2) / 2) / padded_h
        boxes[:, 2] = (x2 - x1) / padded_w
        boxes[:, 3] = (y2 - y1) / padded_h
        
        targets = torch.zeros((len(boxes), 6))
        targets[:, 2:] = boxes[:,0:4]
        targets[:,1] = boxes[:,4]
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, boxes = horisontal_flip(img, boxes)
        return img_path, img, targets
       

    def __len__(self):
        return len(self.ids)

   
    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets