import os
import shutil
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.preprocess import data_transforms

def split_dataset(raw_dir, output_dir, train=0.7, val=0.15, test=0.15):

    classes = os.listdir(raw_dir)

    for cls in classes:
        class_path = os.path.join(raw_dir, cls)

        images = [img for img in os.listdir(class_path)
                  if img.endswith(('.jpg', '.png', '.jpeg'))]

        random.shuffle(images)

        # train / temp
        train_imgs, temp_imgs = train_test_split(
            images, test_size=(1 - train), random_state=42
        )

        # val / test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=test / (test + val),
            random_state=42
        )

        # copy function
        def copy_files(files, split):
            for f in files:
                src = os.path.join(class_path, f)
                dst = os.path.join(output_dir, split, cls, f)

                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")
        copy_files(test_imgs, "test")

    print(" Dataset split completed!")



def get_datasets(data_path):

    train_dataset = ImageFolder(
        root=f"{data_path}/train",
        transform=data_transforms['train']
    )

    val_dataset = ImageFolder(
        root=f"{data_path}/validation",
        transform=data_transforms['val']
    )

    test_dataset = ImageFolder(
        root=f"{data_path}/test",
        transform=data_transforms['test']
    )

    return train_dataset, val_dataset, test_dataset


def get_sampler(train_dataset):

    train_targets = train_dataset.targets
    class_counts = np.bincount(train_targets)
    print("Class counts:", class_counts)

    class_weights = 1. / class_counts
    class_weights = torch.DoubleTensor(class_weights)

    sample_weights = [class_weights[t] for t in train_targets]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def get_loaders(data_path, batch_size=32):


    train_dataset, val_dataset, test_dataset = get_datasets(data_path)

    sampler = get_sampler(train_dataset)


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader