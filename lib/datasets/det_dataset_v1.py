import torch
import torch.utils.data as data
import numpy as np


class DetDataset(data.Dataset):
    """base class for detection dataset"""
    def __init__(self, root, image_sets, dataset_name=None, transform=None, target_transform=None):
        self.name = dataset_name
        self.data_root = root
        self.image_sets = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.ids = []

    def __getitem__(self, index):
        im, gt, extra = self.pull_item(index)
        return im, gt, extra

    def __len__(self):
        return len(self.ids)

    def _setup(self):
        """_setup self.ids"""
        raise NotImplementedError

    def _pre_process(self, index):
        """
        :param index: image id
        :return: img: np.array that's read from opencv;
                target: normalized gt [[xmin, ymin, xmax, ymax, label_ind],]
                extra: extra information, current (height, width, channle)
        """
        raise NotImplementedError

    def pull_item(self, index, tb_writer=None):
        img, target, extra = self._pre_process(index)
        height, width, channels = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            if target.size == 0:
                img, boxes, labels = self.transform(img, None, None, tb_writer)  # target remains as is
            else:
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4], tb_writer)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            img = img[:, :, (2, 1, 0)]  # to rgb
        return torch.from_numpy(img).permute(2, 0, 1), target, extra

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.data_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    targets = []
    extras = []
    for sample in batch:
        imgs.append(sample[0])
        if sample[1].shape==(0,):
            targets.append(torch.FloatTensor([-1]))
        else:
            targets.append(torch.FloatTensor(sample[1]))
        extras.append(torch.FloatTensor(sample[2]))

    extra = torch.stack(extras, 0)
    return torch.stack(imgs, 0), targets, extra
