import os
from .segmentation_list import SegmentationList
from .._util import download as download_data


class Cityscapes(SegmentationList):


    CLASSES = ['back','file']

    ID_TO_TRAIN_ID = {
        0: 0,1: 1 
    }
    TRAIN_ID_TO_COLOR = [(0, 0, 0),(255, 255, 255)]

    EVALUATE_CLASSES = CLASSES

    def __init__(self, root, split='train', data_folder='leftImg8bit', label_folder='gtFine', **kwargs):
        assert split in ['train', 'val']

        # download meta information from Internet
        # list(map(lambda args: download_data(root, *args), self.download_list))
        data_list_file = os.path.join(root, "image_list", "{}.txt".format(split))
        self.split = split
        super(Cityscapes, self).__init__(root, Cityscapes.CLASSES, data_list_file, data_list_file,
                                         os.path.join(data_folder, split), os.path.join(label_folder, split),
                                         id_to_train_id=Cityscapes.ID_TO_TRAIN_ID,
                                         train_id_to_color=Cityscapes.TRAIN_ID_TO_COLOR, **kwargs)

    def parse_label_file(self, label_list_file):
        with open(label_list_file, "r") as f:
            label_list = [line.strip().replace("leftImg8bit", "gtFine_labelIds") for line in f.readlines()]
        return label_list


class FoggyCityscapes(Cityscapes):
    """`Foggy Cityscapes <https://www.cityscapes-dataset.com/>`_ is a real-world semantic segmentation dataset collected
    in foggy driving scenarios.

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, or ``val``.
        data_folder (str, optional): Sub-directory of the image. Default: 'leftImg8bit'.
        label_folder (str, optional): Sub-directory of the label. Default: 'gtFine'.
        beta (float, optional): The parameter for foggy. Choices includes: 0.005, 0.01, 0.02. Default: 0.02
        mean (seq[float]): mean BGR value. Normalize the image if not None. Default: None.
        transforms (callable, optional): A function/transform that  takes in  (PIL image, label) pair \
            and returns a transformed version. E.g, :class:`~common.vision.transforms.segmentation.Resize`.

    .. note:: You need to download Cityscapes manually.
        Ensure that there exist following files in the `root` directory before you using this class.
        ::
            leftImg8bit_foggy/
                train/
                val/
                test/
            gtFine/
                train/
                val/
                test/
    """
    def __init__(self, root, split='train', data_folder='leftImg8bit_foggy', label_folder='gtFine', beta=0.02, **kwargs):
        assert beta in [0.02, 0.01, 0.005]
        self.beta = beta
        super(FoggyCityscapes, self).__init__(root, split, data_folder, label_folder, **kwargs)

    def parse_data_file(self, file_name):
        """Parse file to image list

        Args:
            file_name (str): The path of data file

        Returns:
            List of image path
        """
        with open(file_name, "r") as f:
            data_list = [line.strip().replace("leftImg8bit", "leftImg8bit_foggy_beta_{}".format(self.beta)) for line in f.readlines()]
        return data_list
