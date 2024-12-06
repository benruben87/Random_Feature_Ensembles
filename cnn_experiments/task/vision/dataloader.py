# <codecell>
import glob
import os
import random
from typing import List

import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    ImageMixup,
    LabelMixup,
    RandomHorizontalFlip,
    MixupToOneHot,
    ToDevice,
    ToTensor,
    NormalizeImage,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze

# import sys
# sys.path.append('../')
try:
    from .data_stats import *
except:
    import sys
    sys.path.append('../')

    from data_stats import *


# Define an ffcv dataloader
def get_loader(
    dataset,
    bs,
    mode,
    augment,
    indices=None,
    data_resolution=None,
    crop_resolution=None,
    crop_ratio=(0.75, 1.3333333333333333),
    crop_scale=(0.08, 1.0),
    mixup=0,
    data_path='./beton',
):
    os_cache = OS_CACHED_DICT[dataset]

    if data_resolution is None:
        data_resolution = DEFAULT_RES_DICT[dataset]
    if crop_resolution is None:
        crop_resolution = data_resolution

    real = '' if dataset != 'imagenet_real' or mode == 'train' else 'real_'

    beton_path = os.path.join(
        data_path,
        DATA_DICT[dataset],
        mode,
        real + f'{mode}_{data_resolution}.beton'
    )

    print(f'Loading {beton_path}')

    if dataset == 'imagenet_real' and mode != 'train':
        label_pipeline: List[Operation] = [NDArrayDecoder()]
    else:
        label_pipeline: List[Operation] = [IntDecoder()]

    if augment:
        image_pipeline: List[Operation] = [
            RandomResizedCropRGBImageDecoder((crop_resolution, crop_resolution), ratio=crop_ratio, scale=crop_scale),
            RandomHorizontalFlip(),
        ]
    else:
        image_pipeline: List[Operation] = [
            CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1)
        ]

    # Add image transforms and normalization
    # if mode == 'train' and augment and mixup > 0:
    #     image_pipeline.extend([ImageMixup(alpha=mixup, same_lambda=True)])
    #     label_pipeline.extend([LabelMixup(alpha=mixup, same_lambda=True)])

    return Loader(
        beton_path,
        batch_size=bs,
        num_workers=16,
        order=OrderOption.QUASI_RANDOM,
        drop_last=(mode == 'train'),
        pipelines={'image': image_pipeline, 'label': label_pipeline},
        os_cache=os_cache,
        indices=indices,
    )

if __name__ == '__main__':
    rng = np.random.default_rng(3)
    # idxs = rng.integers(50000, size=1024)
    # idxs = np.random.permutation(np.arange(100))
    idxs = rng.choice(10000, size=1024, replace=False)
    # print('IDXS', idxs)
    # loader = get_loader('cifar100', bs=1024, mode='train', augment=True, mixup=0.8, data_resolution=32, indices=idxs)
    loader = get_loader('cifar100', bs=1024, mode='test', augment=True, mixup=0, data_resolution=32, indices=idxs)
    loader

    import matplotlib.pyplot as plt

    it = iter(loader)
    xs, ys = next(it)

    # xs = torch.permute(xs, (0, 2, 3, 1))

    # plt.imshow(xs[0])
    # xs.shape
    print(ys)

# %%
