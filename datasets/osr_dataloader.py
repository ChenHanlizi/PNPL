import os
import torch
import numpy as np
import pandas as pd
import json
import math
import re
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torch.utils.data import  Dataset
import random
from torchvision.transforms import functional as F
from torch import nn

BICUBIC = Image.BICUBIC
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

class MNIST_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        
        pin_memory = True if use_gpu else False

        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        # print(len(known))
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)
    
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'output'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'output'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

# 这段是原本的
class ImageNetDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        jpeg_path = self.data[index]
        labels = self.labels[index]
        image = Image.open(jpeg_path).convert('RGB')
    
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.as_tensor(int(labels), dtype = torch.int64)
        # print(image.shape)
        return image, labels
    
    def remove_negative_label(self):
        self.data = self.data[self.labels > -1]
        self.labels = self.labels[self.labels > -1]

    def remain_negative_label(self):
        self.data = self.data[self.labels == -2]
        self.labels = self.labels[self.labels == -2]

        
    def __len__(self):
        return len(self.data)
    
class ImageNetDataset_mask(Dataset):
    def __init__(self, data, labels, patch_size=16, ratio=0.99, transform=None):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.ratio = ratio
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        jpeg_path = self.data[index]
        labels = self.labels[index]
        image = Image.open(jpeg_path).convert('RGB')

        # 创建掩码
        mask = self.create_mask(image)

        # 应用掩码
        image = self.apply_mask(image, mask)

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(int(labels), dtype=torch.int64)
        print(image)
        return image, labels

    def create_mask(self, image):
        width, height = image.size
        mask = torch.ones((height, width), dtype=torch.float32)

        num_patches_width = width // self.patch_size
        num_patches_height = height // self.patch_size

        total_patches = num_patches_width * num_patches_height
        num_patches_to_remove = int(total_patches * self.ratio)

        patches_to_remove = random.sample(range(total_patches), num_patches_to_remove)

        for patch_idx in patches_to_remove:
            patch_x = (patch_idx % num_patches_width) * self.patch_size
            patch_y = (patch_idx // num_patches_width) * self.patch_size
            mask[patch_y:patch_y + self.patch_size, patch_x:patch_x + self.patch_size] = 0.0

        return mask

    def apply_mask(self, image, mask):
        image = F.to_tensor(image)
        image = image * mask.unsqueeze(0)  # Unsqueeze mask to match image dimensions
        image = F.to_pil_image(image)
        # print(image)
        return image
    
    
    def remove_negative_label(self):
        self.data = self.data[self.labels > -1]
        self.labels = self.labels[self.labels > -1]

    def remain_negative_label(self):
        self.data = self.data[self.labels == -2]
        self.labels = self.labels[self.labels == -2]

import torch.nn as nn
class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        # x shape: (num_patches, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        return attn_output, attn_weights
class ImageNetDataset1(Dataset):
    def __init__(self, data, labels, transform=None, embed_dim=768, num_heads=8, prob=0.3):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.prob = prob
        self.attention = MultiheadSelfAttention(embed_dim, num_heads)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        jpeg_path = self.data[index]
        labels = self.labels[index]
        image = Image.open(jpeg_path).convert('RGB')
    
        if self.transform is not None:
            image = self.transform(image)
        
        # Assuming image shape is [3, 224, 224]
        image = self.apply_attention_mask(image)
        labels = torch.as_tensor(int(labels), dtype=torch.int64)
        # print(image.shape)
        return image, labels

    def apply_attention_mask(self, image):
        patch_size = 16
        num_patches = 14
        mask_ratio = self.prob

        image_np = image.numpy().transpose(1, 2, 0)  # Convert to numpy array and transpose to [224, 224, 3]
        height, width, channels = image_np.shape

        # Ensure image dimensions are compatible with patch size
        assert height == patch_size * num_patches and width == patch_size * num_patches, "Image size is not 224x224"

        # Reshape image into patches
        image_patches = image_np.reshape(num_patches, patch_size, num_patches, patch_size, channels)
        image_patches = image_patches.swapaxes(1, 2).reshape(-1, patch_size * patch_size * channels)

        # Convert patches to tensor and add batch dimension
        image_patches = torch.tensor(image_patches, dtype=torch.float32).unsqueeze(1)  # Shape: (num_patches, 1, embed_dim)

        # Apply multihead self-attention
        attn_output, attn_weights = self.attention(image_patches)

        # Compute attention scores and select patches to mask
        attn_scores = attn_weights.mean(dim=1).squeeze(1)  # Shape: (num_patches)
        # print('注意力分数',attn_scores)
        num_masked_patches = int(attn_scores.size(0) * mask_ratio)
        _, masked_indices = torch.topk(attn_scores, num_masked_patches, largest=False)

        # Mask selected patches
        image_patches[masked_indices] = 0

        # Reshape patches back to original image shape
        image_patches = image_patches.squeeze(1).numpy().reshape(num_patches, num_patches, patch_size, patch_size, channels)
        image_np = image_patches.swapaxes(1, 2).reshape(height, width, channels)

        image_np = image_np.transpose(2, 0, 1)  # Convert back to [3, 224, 224]
        return torch.tensor(image_np, dtype=torch.float32)

    def remove_negative_label(self):
        mask = self.labels > -1
        self.data = [self.data[i] for i in range(len(mask)) if mask[i]]
        self.labels = self.labels[mask]

    def remain_negative_label(self):
        mask = self.labels == -2
        self.data = [self.data[i] for i in range(len(mask)) if mask[i]]
        self.labels = self.labels[mask]

    def __len__(self):
        return len(self.data)



class ImageNet1K_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols/imagenet_class_index.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'ImageNet1K', 'protocols', 'imagenet_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"ImageNet1k_p1":'p1', 'ImageNet1k_p2':'p2', 'ImageNet1k_p3':'p3', 'ImageNet1k_p4':'p4', 'ImageNet1k_p5':'p5', 'ImageNet1k_p6':'p6',
                        "ImageNet1k_p7":'p7', 'ImageNet1k_p8':'p8', 'ImageNet1k_p9':'p9', 'ImageNet1k_p10':'p10', 'ImageNet1k_p11':'p11', 'ImageNet1k_p12':'p12'}
        train_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        known = [filedir_name[fileid] for fileid in key_list]
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 10)]
        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]

        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]
        if datasplit in ["ImageNet1k_p1", "ImageNet1k_p2", "ImageNet1k_p3"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        # test_set_open_vocabulary = test_set[[name_dic[x][0] for x in test_set.iloc[:,1]].isin(key_list_open_vocabulary)]
        
        image_root = dataroot + '/ImageNet1K/ILSVRC/Data/CLS-LOC/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)
        
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        print('All Train Data:', len(Trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        print('All Test Data:', len(Testset))
        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(Trainset), 'Test: ', len(Testset), 'Out: ', len(Outset))
        print('All Test: ', (len(Testset) + len(Outset)))
        
        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary



class AID_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'AID', 'aid_class_index_unknown14.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'AID','aid_unknown14_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"AID_p1":'p1',"AID_p2":'p2', "AID_p3":'p3'}
        train_file_path = os.path.join(dataroot, 'AID', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'AID', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        
        known = [filedir_name[fileid] for fileid in key_list]
        
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 1)]
        
        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]  # key_list
        
        print(known_open_vocabulary)
       
        print(train_set.iloc[:,0].str.split('/').str[1])

        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]  
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]   # key list
        
        if datasplit in ["AID_p1","AID_p2","AID_p3"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        
        
        image_root = dataroot + '/AID/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        # test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")

        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")


        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)
        
        # Trainset = ImageNetDataset_mask(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), patch_size=16, ratio=0.1, transform=train_transform)
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        
        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary
        


class UCM_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'UCM', 'ucm_class_index_unknown5.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'UCM','ucm_unknown5_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"UCM_p1":'p1',"UCM_p2":'p2', "UCM_p3":'p3'}
        train_file_path = os.path.join(dataroot, 'UCM', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'UCM', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]

        known = [filedir_name[fileid] for fileid in key_list]
        
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 1)]

        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]  # key_list


        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]  
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]   # key list
        
        if datasplit in ["UCM_p1"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]

        
        image_root = dataroot + '/UCM/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        # test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")

        # test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")


        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        # Trainset_open_vocabulary = ImageNetDataset_0(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)

        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)

        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary


class NWPU_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'NWPU', 'nwpu_class_index_unknown11.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'NWPU','nwpu_unknown11_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"NWPU_p1":'p1',"NWPU_p2":'p2', "NWPU_p3":'p3'}
        train_file_path = os.path.join(dataroot, 'NWPU', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'NWPU', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        # print(key_list)
        known = [filedir_name[fileid] for fileid in key_list]
        
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 1)]
        
        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]  # key_list
        

        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]  
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]   # key list
        
        if datasplit in ["NWPU_p1","NWPU_p2","NWPU_p3"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        
        
        image_root = dataroot + '/NWPU/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        # test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        print(train_set_open_vocabulary.iloc[:,0])

        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        print('#########################')
        print(test_set_open_vocabulary.iloc[:,0])

        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)
        
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary

class ImageNet_OOD(object):
    def __init__(self, ood_dataset, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, shot = 0):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols/imagenet_class_index.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'ImageNet1K', 'protocols', 'imagenet_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1   
        train_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols',  'ood_train.csv')
        test_file_path = os.path.join(dataroot, 'ImageNet1K', 'protocols',  'ood_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        known = [filedir_name[fileid] for fileid in key_list]
        
        image_root = dataroot + '/ImageNet1K/ILSVRC/Data/CLS-LOC/'
        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        FewshotSampler = FewshotRandomSampler(Trainset, num_samples_per_class=shot)
        print("ood_dataset = ", ood_dataset)
        if ood_dataset == 'iNaturalist':
            Outset = ImageFolder(root=os.path.join(dataroot, 'iNaturalist'), transform=transform)
        elif ood_dataset == 'SUN':
            Outset = ImageFolder(root=os.path.join(dataroot, 'SUN'), transform=transform)
        elif ood_dataset == 'places365': # filtered places
            Outset = ImageFolder(root= os.path.join(dataroot, 'Places'),transform=transform)  
        elif ood_dataset == 'dtd':
            Outset = ImageFolder(root=os.path.join(dataroot, 'dtd', 'images'),transform=transform)
        elif ood_dataset == 'NINCO':
            Outset = ImageFolder(root=os.path.join(dataroot, 'NINCO', 'NINCO_OOD_classes'),transform=transform)
        elif ood_dataset == 'ImageNet-O':
            Outset = ImageFolder(root=os.path.join(dataroot, 'imagenet-o'),transform=transform)
        elif ood_dataset == 'ImageNet-1K-OOD':
            Outset = ImageFolder(root=os.path.join(dataroot, 'imagenet-1k-ood'), transform=transform)
        elif ood_dataset == 'OpenImage-O':
            Outset = ImageFolder(root=os.path.join(dataroot, 'OpenImage-O'), transform=transform)
        else:
            print("Wrong ood dataset!!")
        
        self.num_classes = len(known)
        self.known = known
        pin_memory = True if use_gpu else False

        print('All Train Data:', len(Trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler
        )
        
        print('All Test Data:', len(Testset))
        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(Trainset), 'Test: ', len(Testset), 'Out: ', len(Outset))
        print('All Test: ', (len(Testset) + len(Outset)))

class FewshotRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_samples_per_class):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.class_counts = np.bincount(self.labels)
        self.num_samples_per_class = num_samples_per_class
        self.indices = self._get_indices()
        
    def _get_indices(self):
        indices = []
        for class_label in np.unique(self.labels):
            class_indices = np.where(self.labels == class_label)[0]
            if self.num_samples_per_class <= 0:
                # print(self.class_counts[class_label])
                class_indices = np.random.choice(class_indices, size=self.class_counts[class_label], replace=False) 
            else:
                class_indices = np.random.choice(class_indices, size=self.num_samples_per_class, replace=False)
            
            indices.extend(class_indices.tolist())
        indices = np.random.permutation(indices)
        return indices
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class tiny1_OSR(object):
    def __init__(self, datasplit, dataroot='./data', use_gpu=True, num_workers=8, batch_size=128, img_size=224, few_shot = 0, cfg = None):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ])

        json_file_path = os.path.join(dataroot, 'tiny_imagenet', 'tiny_imagenet_class_index.json')
        with open(json_file_path, 'r') as f:
            name_dic = json.load(f)
        clean_names = np.load(os.path.join(dataroot, 'tiny_imagenet','tiny_imagenet_class_clean.npy'))
        filedir_name = {}
        t_for_clean_change = 0
        for k, v in name_dic.items():
            filedir_name[v[0]] = clean_names[t_for_clean_change]
            t_for_clean_change += 1
        csv_name_dic = {"tiny1_p3":'p3' ,"AID_p4":'p4'}
        train_file_path = os.path.join(dataroot, 'tiny_imagenet', csv_name_dic[datasplit] + '_train.csv')
        test_file_path = os.path.join(dataroot, 'tiny_imagenet', csv_name_dic[datasplit] + '_test.csv')
        train_set = pd.read_csv(train_file_path)
        test_set = pd.read_csv(test_file_path)
        
        dir_label = train_set.groupby(train_set.iloc[:,0].str.split('/').str[1]).first().iloc[:,1].to_dict()
        key_list = [k for k, v in sorted(dir_label.items(), key=lambda item: item[1]) if v > -1]
        # print(key_list)
        known = [filedir_name[fileid] for fileid in key_list]
        # 这里是只选十分之一类吗？
        key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 10)]
        # key_list_open_vocabulary = key_list[:math.ceil(len(key_list) // 2)]
        # print(key_list_open_vocabulary)
        known_open_vocabulary = [filedir_name[fileid] for fileid in key_list_open_vocabulary]
        # print(known_open_vocabulary)
        print('herehere')
        print(train_set.iloc[:,0].str.split('/').str[1])

        train_set_open_vocabulary = train_set[train_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        # print(train_set_open_vocabulary)
        all_test_set_names = []
        for x in test_set.iloc[:,1]:
            if x < 0:
                all_test_set_names.append("nothing")
            else:
                all_test_set_names.append(name_dic[str(x)][0])
        all_test_set_names = pd.Series(all_test_set_names)
        test_set_open_vocabulary = test_set[all_test_set_names.isin(key_list_open_vocabulary)]
        # 这里应该用不到
        if datasplit in ["tiny1_p3"]:
            test_set_open_vocabulary = test_set[test_set.iloc[:,0].str.split('/').str[1].isin(key_list_open_vocabulary)]
        # test_set_open_vocabulary = test_set[[name_dic[x][0] for x in test_set.iloc[:,1]].isin(key_list_open_vocabulary)]
        
        image_root = dataroot + '/tiny_imagenet/'

        train_set.iloc[:,0] = train_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        # test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set.iloc[:,0] = test_set.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        
        train_set_open_vocabulary.iloc[:,0] = train_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        print(train_set_open_vocabulary.iloc[:,0])
        # test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{re.sub(r'n[0-9]*/','', x)}")
        test_set_open_vocabulary.iloc[:,0] = test_set_open_vocabulary.iloc[:,0].apply(lambda x: f"{image_root}{x}")
        print('#########################')
        print(test_set_open_vocabulary.iloc[:,0])
        Trainset_open_vocabulary = ImageNetDataset(np.array(train_set_open_vocabulary.iloc[:,0]), np.array(train_set_open_vocabulary.iloc[:,1]), transform=train_transform)
        Testset_open_vocabulary = ImageNetDataset(np.array(test_set_open_vocabulary.iloc[:,0]), np.array(test_set_open_vocabulary.iloc[:,1]), transform=transform)
        # print(train_set_open_vocabulary)
        # print(test_set_open_vocabulary)
        
        Trainset = ImageNetDataset(np.array(train_set.iloc[:,0]), np.array(train_set.iloc[:,1]), transform=train_transform)
        Testset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        Outset = ImageNetDataset(np.array(test_set.iloc[:,0]), np.array(test_set.iloc[:,1]), transform=transform)
        
        Trainset_open_vocabulary.remove_negative_label()
        Testset_open_vocabulary.remove_negative_label()
        Trainset.remove_negative_label()
        Testset.remove_negative_label()
        Outset.remain_negative_label()
        
        FewshotSampler_Trainset = FewshotRandomSampler(Trainset, num_samples_per_class=few_shot)
        FewshotSampler_Trainset_open_vocabulary = FewshotRandomSampler(Trainset_open_vocabulary, num_samples_per_class=few_shot)
    
        self.num_classes = len(known)
        self.known = known
        self.classes = known
        
        self.num_classes_open_vocabulary = len(known_open_vocabulary)
        self.known_open_vocabulary = known_open_vocabulary
        self.classes_open_vocabulary = known_open_vocabulary
        
        print('Selected Labels: ', known)

        pin_memory = True if use_gpu else False

        print('All Train Data:', len(Trainset))
        
        self.train_loader = torch.utils.data.DataLoader(
            Trainset, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset
        )
        
        self.train_loader_open_vocabulary = torch.utils.data.DataLoader(
            Trainset_open_vocabulary, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, sampler=FewshotSampler_Trainset_open_vocabulary
        )
        self.test_loader_open_vocabulary = torch.utils.data.DataLoader(
            Testset_open_vocabulary, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        print('All Test Data:', len(Testset))
        self.test_loader = torch.utils.data.DataLoader(
            Testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.out_loader = torch.utils.data.DataLoader(
            Outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(Trainset), 'Test: ', len(Testset), 'Out: ', len(Outset))
        print('All Test: ', (len(Testset) + len(Outset)))
        
        if cfg['stage'] <= 2:
            self.train_loader = self.train_loader_open_vocabulary
            self.test_loader = self.test_loader_open_vocabulary
            self.num_classes = self.num_classes_open_vocabulary
            self.known = self.known_open_vocabulary
            self.classes = self.classes_open_vocabulary
