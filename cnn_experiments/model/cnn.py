"""
Simple MLP model
"""

from collections.abc import Callable, Sequence
from dataclasses import field
from functools import partial
from typing import Any 

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

ModuleDef = Any

def parse_act_fn(fn: str):
    if fn == 'relu':
        return jax.nn.relu
    elif fn == 'linear':
        return lambda x: x
    elif fn == 'gelu':
        return jax.nn.gelu
    elif fn =='quadratic':
        return lambda x: x**2
    else:
        raise ValueError(f'function not recognized: {fn}')


@struct.dataclass
class CnnConfig:
    """Global hyperparamters"""
    cnn_widths: list = field(default_factory=list)
    mlp_widths: list = field(default_factory=list)
    n_out: int = 1
    act_fn: str = 'relu'
    freeze_intermediate_layers: bool = False

    def to_model(self):
        return CNN(self)


class CNN(nn.Module):

    config: CnnConfig

    @nn.compact
    def __call__(self, x):
        act_fn = parse_act_fn(self.config.act_fn)

        for i, width in enumerate(self.config.cnn_widths):
            name = None
            if self.config.freeze_intermediate_layers:
                name = f'Conv_{i}_freeze'

            x = nn.Conv(features=width, kernel_size=(5, 5), strides=2, name=name)(x)
            x = act_fn(x)
        
        x = x.reshape((x.shape[0], -1))
        
        for width in self.config.mlp_widths:
            name = None
            if self.config.freeze_intermediate_layers:
                name = f'Dense_{i}_freeze'

            x = nn.Dense(features=width, name=name)(x)
            x = act_fn(x)

        x = nn.Dense(features=self.config.n_out)(x)

        return x


@struct.dataclass
class CnnEnsembleConfig:
    """Global hyperparamters"""
    member_config: CnnConfig
    n_members: int = 3

    def to_model(self):
        return CnnEnsemble(self)


class CnnEnsemble(nn.Module):

    config: CnnEnsembleConfig

    @nn.compact
    def __call__(self, x):
        ys = [CNN(self.config.member_config)(x) for _ in range(self.config.n_members)]
        return jnp.stack(ys, axis=0)
        

@struct.dataclass
class ResNetConfig:
    """Global hyperparamters"""
    stage_sizes: Sequence[int]
    n_out: int
    n_features: int = 64
    freeze_intermediate_layers: bool = False

    def to_model(self):
        return ResNet(self.stage_sizes, self.n_out, num_filters=self.n_features, freeze_intermediate_layers=self.freeze_intermediate_layers)


@struct.dataclass
class EnsembleConfig:
    member_config: ModuleDef
    n_members: int = 3

    def to_model(self):
        return Ensemble(self)


class Ensemble(nn.Module):

    config: ModuleDef

    @nn.compact
    def __call__(self, x):
        ys = [self.config.member_config.to_model()(x) for _ in range(self.config.n_members)]
        return jnp.stack(ys, axis=0)


# ResNet implementation adapted from https://github.com/google/flax
class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1.5."""

  stage_sizes: Sequence[int]
  num_classes: int
  num_filters: int = 64
  block_cls: ModuleDef = ResNetBlock
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  freeze_intermediate_layers: bool = False

  @nn.compact
  def __call__(self, x):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    # norm = partial(
    #     nn.BatchNorm,
    #     use_running_average=not train,
    #     momentum=0.9,
    #     epsilon=1e-5,
    #     dtype=self.dtype,
    #     axis_name='batch',
    # )
    norm = partial(nn.LayerNorm)  # drop in for convenience

    name = 'conv_init'
    if self.freeze_intermediate_layers:
        name = 'conv_init_freeze'
        
    x = conv(
        self.num_filters,
        # (7, 7),
        (3, 3),
        (2, 2),
        # padding=[(3, 3), (3, 3)],
        padding=[(1, 1), (1, 1)],
        name=name,
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    # x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)

        name = None if self.freeze_intermediate_layers == False else f'ResNetBlock_{i + j}_freeze'
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
            name=name
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x