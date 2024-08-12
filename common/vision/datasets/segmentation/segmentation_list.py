import os
from typing import Sequence, Optional, Dict, Callable
from PIL import Image
import tqdm
import numpy as np
from torch.utils import data
import torch
import colorsys
import cv2
class SegmentationList(data.Dataset):

    def __init__(self, root: str, classes: Sequence[str], data_list_file: str, label_list_file: str,
                 data_folder: str, label_folder: str,
                 id_to_train_id: Optional[Dict] = None, train_id_to_color: Optional[Sequence] = None,
                 transforms: Optional[Callable] = None):
        self.root = root
        self.classes = classes
        self.data_list_file = data_list_file
        self.label_list_file = label_list_file
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.ignore_label = 255
        self.id_to_train_id = id_to_train_id
        self.train_id_to_color = np.array(train_id_to_color)
        self.data_list = self.parse_data_file(self.data_list_file)
        self.label_list = self.parse_label_file(self.label_list_file)
        self.transforms = transforms

    def parse_data_file(self, file_name):
        """Parse file to image list

        Args:
            file_name (str): The path of data file

        Returns:
            List of image path
        """
        with open(file_name, "r") as f:
            data_list = [line.strip() for line in f.readlines()]
        return data_list

    def parse_label_file(self, file_name):

        with open(file_name, "r") as f:
            label_list = [line.strip() for line in f.readlines()]
        return label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_name = self.data_list[index]
        label_name = self.label_list[index]
        image = Image.open(os.path.join(self.root, self.data_folder, image_name)).convert('RGB')
        image = np.array(image)


        luv=cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        yuv=cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        u1 = luv[:,:,1]
        v2 = yuv[:,:,2]
        s = hsv[:,:,1]
        image = np.dstack((u1, v2, s))        

        # 
        image = Image.fromarray(image) 

        # image = Image.fromarray(np.uint32(image))
        # print(image)
        label = Image.open(os.path.join(self.root, self.label_folder, label_name))
        image, label = self.transforms(image, label)

        # remap label
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        label = np.asarray(label, np.int64)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.int64)
        if self.id_to_train_id:
            for k, v in self.id_to_train_id.items():
                label_copy[label == k] = v

        return image, label_copy.copy()

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    def decode_target(self, target):

        target = target.copy()
        target[target == 255] = self.num_classes # unknown label is black on the RGB label
        target = self.train_id_to_color[target]
        return Image.fromarray(target.astype(np.uint8))

    def collect_image_paths(self):
        """Return a list of the absolute path of all the images"""
        return [os.path.join(self.root, self.data_folder, image_name) for image_name in self.data_list]

    @staticmethod
    def _save_pil_image(image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)

    def translate(self, transform: Callable, target_root: str, color=False):

        os.makedirs(target_root, exist_ok=True)
        for image_name, label_name in zip(tqdm.tqdm(self.data_list), self.label_list):
            image_path = os.path.join(target_root, self.data_folder, image_name)
            label_path = os.path.join(target_root, self.label_folder, label_name)
            if os.path.exists(image_path) and os.path.exists(label_path):
                continue
            image = Image.open(os.path.join(self.root, self.data_folder, image_name)).convert('RGB')
            label = Image.open(os.path.join(self.root, self.label_folder, label_name))

            translated_image, translated_label = transform(image, label)
            self._save_pil_image(translated_image, image_path)
            self._save_pil_image(translated_label, label_path)
            if color:
                colored_label = self.decode_target(np.array(translated_label))
                file_name, file_ext = os.path.splitext(label_name)
                self._save_pil_image(colored_label, os.path.join(target_root, self.label_folder,
                                                                 "{}_color{}".format(file_name, file_ext)))

    @property
    def evaluate_classes(self):
        """The name of classes to be evaluated"""
        return self.classes

    @property
    def ignore_classes(self):
        """The name of classes to be ignored"""
        return list(set(self.classes) - set(self.evaluate_classes))