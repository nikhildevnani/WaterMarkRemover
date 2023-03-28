import os
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.remove_uncommon_files(root)
        self.transform = transform
        self.without_annotations_dir = os.path.join(self.root, "no-watermark")
        self.with_annotations_dir = os.path.join(self.root, "watermark")
        # Get the list of image filenames
        self.filenames = os.listdir(self.with_annotations_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load the image and convert it to a tensor
        with_annotation_image = Image.open(os.path.join(self.with_annotations_dir, self.filenames[idx]))
        without_annotation_image = Image.open(os.path.join(self.without_annotations_dir, self.filenames[idx]))
        with_annotation_image = F.to_tensor(with_annotation_image)
        without_annotation_image = F.to_tensor(without_annotation_image)

        # Apply the specified transformations
        if self.transform:
            with_annotation_image = self.transform(with_annotation_image)
            without_annotation_image = self.transform(without_annotation_image)


        # Return the image and its index
        return with_annotation_image, without_annotation_image, idx

    def remove_uncommon_files(self, root):
        without_annotations = os.path.join(root, "no-watermark")
        with_annotations = os.path.join(root, "watermark")
        with_annotations_filenames = set(os.listdir(with_annotations))
        without_annotations_filenames = set(os.listdir(without_annotations))
        common_files = without_annotations_filenames.intersection(with_annotations_filenames)
        total_files = with_annotations_filenames.union(without_annotations_filenames)
        uncommon_files = total_files-common_files
        for file in uncommon_files:
            with_annotation_path = os.path.join(with_annotations, file)
            without_annotation_path = os.path.join(without_annotations, file)
            if os.path.exists(with_annotation_path):
                os.remove(with_annotation_path)
            if os.path.exists(without_annotation_path):
                os.remove(without_annotation_path)