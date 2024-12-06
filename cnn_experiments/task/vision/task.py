"""
Computer vision tasks
"""
# <codecell>
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

try:
    from .dataloader import get_loader
    from .data_stats import *
except:
    import sys
    sys.path.append('../')

    from dataloader import get_loader
    from data_stats import *


# adapted from https://docs.ffcv.io/_modules/ffcv/transforms/mixup.html#MixupToOneHot
@functools.partial(jax.jit, static_argnames=['batch_size', 'n_classes'])
def mixup_to_onehot_and_smooth(mix, batch_size, n_classes, label_smooth):
    N = batch_size
    dst = jnp.zeros((N, n_classes))
    dst = dst.at[jnp.arange(N), mix[:, 0].astype(int)].set(mix[:, 2])
    mix = mix.at[:, 2].set(1 - mix[:, 2])
    dst = dst.at[jnp.arange(N), mix[:, 1].astype(int)].set(mix[:, 2])
    return optax.smooth_labels(dst, alpha=label_smooth)


class Vision:
    def __init__(self, 
                 name, 
                 data_path=None,
                 data_resolution=32, 
                 augment=True, mixup=0, label_smooth=0, 
                 mode='train', data_size=None,
                 batch_size=128, loop=False) -> None:

        self.data_path = data_path
        if self.data_path is None:
            self.data_path = Path(__file__).parent / 'beton'

        self.name = name
        self.data_resolution = data_resolution
        self.augment = augment
        self.mixup = mixup
        self.label_smooth = label_smooth
        self.mode = mode
        self.data_size = data_size
        self.batch_size = batch_size
        self.loop = loop

        self.mean = MEAN_DICT[name]
        self.std = STD_DICT[name]
        self.n_classes = CLASS_DICT[name]

        if mode == 'train':
            self.n_examples = SAMPLE_DICT[name]
        elif mode == 'test':
            self.n_examples = SAMPLE_DICT_TEST[name]
        else:
            raise ValueError(f'mode unrecognized: {mode}')

        self.idxs = None
        if self.data_size is not None:
            assert self.data_size <= self.n_examples, f'data_size of {self.data_size} > {self.n_examples} total examples'
            self.idxs = np.random.choice(self.n_examples, size=self.data_size, replace=False)

        self.loader = get_loader(name, bs=self.batch_size, mode=self.mode, augment=self.augment, mixup=self.mixup, data_resolution=self.data_resolution, data_path=self.data_path, indices=self.idxs)
        self.it = iter(self.loader)

    def __next__(self):
        try:
            xs, ys = next(self.it)
            xs = (xs - self.mean) / self.std

            if self.mixup > 0:
                ys = mixup_to_onehot_and_smooth(ys, ys.shape[0], self.n_classes, self.label_smooth)
            else:
                ys = jax.nn.one_hot(ys.squeeze(), self.n_classes)

            return xs, ys

        except StopIteration:
            if self.loop:
                return next(iter(self))
            else:
                raise StopIteration


    def __iter__(self):
        self.it = iter(self.loader)
        return self


# <codecell>
if __name__ == '__main__':
    task = Vision('cifar10', data_resolution=32, augment=False, batch_size=5, mode='train', data_size=None)

    # for _ in range(3):
    #     for batch in task:
    #         print('iter')

    xs, ys = next(task)
    print(ys.shape)

    # from tqdm import tqdm

    # for i in tqdm(range(3 * len(task.loader))):
    #     xs, ys = next(task)

    #     if i % len(task.loader) == 0:
    #         print(ys[0])


# %%
    import matplotlib.pyplot as plt
    plt.imshow(xs[4])
    print(ys.argmax(axis=-1))

# %%
    xs

# %%
