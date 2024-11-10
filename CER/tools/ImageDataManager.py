import torch
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)

import math
import random

class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125), then
    a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)
                                    ), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )
        return croped_img


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        mean=[0.4914, 0.4822, 0.4465]
    ):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def build_transforms(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    # if 'random_patch' in transforms:
    #     print('+ random patch')
    #     transform_tr += [RandomPatch()]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        load_train_targets (bool, optional): construct train-loader for target datasets.
            Default is False. This is useful for domain adaptation research.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        num_cams (int, optional): number of cameras to sample in a batch (when using
            ``RandomDomainSampler``). Default is 1.
        num_datasets (int, optional): number of datasets to sample in a batch (when
            using ``RandomDatasetSampler``). Default is 1.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        train_sampler_t (str, optional): sampler for target train loader. Default is RandomSampler.
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader

        # return train loader of target data
        train_loader_t = datamanager.train_loader_t
    """
    data_type = 'image'

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler='RandomSampler',
        train_sampler_t='RandomSampler',
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False
    ):

        super(ImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        self.train_loader_t = None
        if load_train_targets:
            # check if sources and targets are identical
            assert len(set(self.sources) & set(self.targets)) == 0, \
                'sources={} and targets={} must not have overlap'.format(self.sources, self.targets)

            print('=> Loading train (target) dataset')
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    transform=self.transform_tr,
                    k_tfm=k_tfm,
                    mode='train',
                    combineall=False, # only use the training data
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    train_sampler_t,
                    batch_size=batch_size_train,
                    num_instances=num_instances,
                    num_cams=num_cams,
                    num_datasets=num_datasets
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source images   : {}'.format(len(trainset)))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        if load_train_targets:
            print(
                '  # target images   : {} (unlabeled)'.format(len(trainset_t))
            )
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')