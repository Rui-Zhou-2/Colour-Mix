from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Normalize


class ResizeImage(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
          (h, w), output size will be matched to this. If size is an int,
          output size will be (size, size)
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class MultipleApply:
    """Apply a list of transformations to an image and get multiple transformed images.

    Args:
        transforms (list or tuple): list of transformations

    Example:
        
        >>> transform1 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224)
        ... ])
        >>> transform2 = T.Compose([
        ...     ResizeImage(256),
        ...     T.RandomCrop(224),
        ... ])
        >>> multiply_transform = MultipleApply([transform1, transform2])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        return [t(image) for t in self.transforms]


class Denormalize(Normalize):
    """DeNormalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will denormalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = input[channel] * std[channel] + mean[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.

    """
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        super().__init__((-mean / std).tolist(), (1 / std).tolist())


class NormalizeAndTranspose:
# RGB   
    # def __init__(self, mean=(104.00698793, 116.66876762, 122.67891434)): 
    # def __init__(self, mean=(93.96824082, 102.91038224, 117.93673099), std=(54.26331842, 50.59739983, 59.18039747)):

# U1V2H
    def __init__(self, mean=(138.1275503, 105.52919886, 164.28542046), std=(22.03597601, 17.27548111, 0.78641089)):

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)  

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.asarray(image, np.float32)

            # image -= self.mean
            # image = image/ self.std
            image = image.transpose((2, 0, 1)).copy()
        # elif isinstance(image, torch.Tensor):

        #     image -= torch.from_numpy(self.mean).to(image.device)
        #     image = image/ torch.from_numpy(self.std)
        #     image = image.permute((2, 0, 1))
        else:
            raise NotImplementedError(type(image))
        
        return image


class DeNormalizeAndTranspose:
# RGB   
    # def __init__(self, mean=(104.00698793, 116.66876762, 122.67891434)):
    # def __init__(self, mean=(93.96824082, 102.91038224, 117.93673099), std=(54.26331842, 50.59739983, 59.18039747)):

    def __init__(self, mean=(138.1275503, 105.52919886, 164.28542046), std=(22.03597601, 17.27548111, 0.78641089)):


        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image):
        image = image.transpose((1, 2, 0))
        # denormalize
        # image = image * self.std
        # image += self.mean

        return image

