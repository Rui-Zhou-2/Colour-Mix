import os
from .segmentation_list import SegmentationList
from .cityscapes import Cityscapes
from .._util import download as download_data


class GTA5(SegmentationList):


    def __init__(self, root, split='train', data_folder='images', label_folder='labels', **kwargs):
        assert split in ['train']
        # download meta information from Internet
        # list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(GTA5, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file, data_folder, label_folder,
                                   id_to_train_id=Cityscapes.ID_TO_TRAIN_ID, train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)