"""
Training utilities
"""

from dataclasses import dataclass, field
from functools import partial
import itertools
from typing import Any, Iterable

from flax import struct, traverse_util
from flax.training import train_state
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

from common import new_seed


@struct.dataclass
class Metrics:
    accuracy: float
    loss: float
    count: int = 0

    @staticmethod
    def empty():
        return Metrics(accuracy=-1, loss=-1)
    
    def merge(self, other):
        total = self.count + 1
        acc = (self.count / total) * self.accuracy + (1 / total) * other.accuracy
        loss = (self.count / total) * self.loss + (1 / total) * other.loss
        return Metrics(acc, loss, count=total)


class TrainState(train_state.TrainState):
    metrics: Metrics
    init_params: Any = None


def create_train_state(rng, model, dummy_input, use_freeze=True, lr=1e-4, optim=optax.adamw, **opt_kwargs):
    params = model.init(rng, dummy_input)['params']
    tx = optim(learning_rate=lr, **opt_kwargs)

    if use_freeze:
        tx = optax.multi_transform(
            {'learn': tx,
            'freeze': optax.set_to_zero()},
            traverse_util.path_aware_map(lambda path, _: 'freeze' if np.any([s.endswith('freeze') for s in path]) else 'learn', params)
        )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics=Metrics.empty(),
        init_params=params
    )

def parse_loss_name(loss):
    loss_func = None
    if loss == 'bce':
        loss_func = optax.sigmoid_binary_cross_entropy
    elif loss == 'ce':
        loss_func = optax.softmax_cross_entropy_with_integer_labels
    elif loss == 'ce_onehot':
        loss_func = optax.softmax_cross_entropy
    elif loss == 'mse':
        loss_func = optax.squared_error
    else:
        raise ValueError(f'unrecognized loss name: {loss}')
    return loss_func


@partial(jax.jit, static_argnames=('gamma', 'loss',))
def train_step(state, batch, gamma=None, loss='bce', beta=None):
    x, labels = batch
    loss_func = parse_loss_name(loss)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)

        if gamma is not None:
            logits_init = state.apply_fn({'params': state.init_params}, x)
            logits = (1 / gamma) * (logits - logits_init)

        if beta is None:
            train_loss = loss_func(logits, labels)

            if loss == 'bce' and len(labels.shape) > 1:
                assert logits.shape == train_loss.shape
                train_loss = train_loss.mean(axis=-1)
        else:
            loss_avg = jnp.mean(loss_func(logits, labels), axis=0)
            score_avg = jnp.mean(logits, axis=0)
            train_loss = (1 - beta) * loss_avg + beta * loss_func(score_avg, labels)

        assert len(train_loss.shape) == 1
        return train_loss.mean()
    
    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@partial(jax.jit, static_argnames=('loss',))
def compute_metrics(state, batch, loss='bce'):
    x, labels = batch
    logits = state.apply_fn({'params': state.params}, x)

    if len(logits.shape) == 3:  # ensembled
        logits = jnp.mean(logits, axis=0)  # score averaged

    loss_func=parse_loss_name(loss)
    loss = loss_func(logits, labels).mean()

    if len(logits.shape) == 1:
        preds = logits > 0
    else:
        preds = logits.argmax(axis=1)
    
    if len(labels.shape) > 1:
        labels = labels.argmax(axis=1)
    
    acc = jnp.mean(preds == labels)

    metrics = Metrics(accuracy=acc, loss=loss)
    metrics = state.metrics.merge(metrics)
    state = state.replace(metrics=metrics)
    return state


def train(config, data_iter,
          test_iter=None, start_params=None,
          loss='ce', gamma=None, beta=None,
          n_epochs=1_000, test_every=100, save_params=False, 
          early_stop_n=None, early_stop_key='loss', early_stop_decision='min',
          optim=optax.adamw, 
          seed=None, 
          **opt_kwargs):

    if seed is None:
        seed = new_seed()
    
    if test_iter is None:
        test_iter = data_iter
    
    init_rng = jax.random.key(seed)
    model = config.to_model()

    samp_x, _ = next(data_iter)
    state = create_train_state(init_rng, model, samp_x, optim=optim, **opt_kwargs)

    if start_params is not None:
        state = state.replace(params=start_params)

    hist = {
        'train': [],
        'test': [],
        'params': []
    }

    for step in tqdm(range(n_epochs)):
        for batch in data_iter:
            state = train_step(state, batch, loss=loss, gamma=gamma, beta=beta)
            state = compute_metrics(state, batch, loss=loss)

        if ((step + 1) % test_every == 0) or ((step + 1) == n_epochs):
            hist['train'].append(state.metrics)

            state = state.replace(metrics=Metrics.empty())
            test_state = state

            for test_batch in test_iter:
                test_state = compute_metrics(test_state, test_batch, loss=loss)
            
            hist['test'].append(test_state.metrics)

            _print_status(step+1, hist)

            if save_params:
                hist['params'].append(state.params)
        
            if early_stop_n is not None and len(hist['train']) > early_stop_n:
                last_n_metrics = np.array([getattr(m, early_stop_key) for m in hist['train'][-early_stop_n - 1:]])
                if early_stop_decision == 'min' and np.all(last_n_metrics[0] < last_n_metrics[1:]) \
                or early_stop_decision == 'max' and np.all(last_n_metrics[0] > last_n_metrics[1:]):
                    print(f'info: stopping early with {early_stop_key} =', last_n_metrics[-1])
                    break
    
    return state, hist

            
def _print_status(step, hist):
    print(f'ITER {step}:  train_loss={hist["train"][-1].loss:.4f}   train_acc={hist["train"][-1].accuracy:.4f}   test_acc={hist["test"][-1].accuracy:.4f}')


@dataclass
class Case:
    name: str
    config: dataclass
    train_task: Iterable = None
    test_task: Iterable = None
    train_args: dict = field(default_factory=dict)
    state: list = None
    hist: list = None
    info: dict = field(default_factory=dict)

    def run(self):
        self.state, self.hist = train(self.config, data_iter=self.train_task, test_iter=self.test_task, **self.train_args)
    
    def get_flops(self):
        train_args = self.train_args
        loss = train_args.get('loss', None)
        return get_flops(train_step, self.state, next(self.train_task), loss=loss)
    
    def eval(self, task, key_name='eval_acc'):
        xs, ys = next(task)
        logits = self.state.apply_fn({'params': self.state.params}, xs)

        if len(logits.shape) > 1:
            preds = logits.argmax(-1)
        else:
            preds = (logits > 0).astype(float)

        eval_acc = np.mean(ys == preds)
        self.info[key_name] = eval_acc
    
    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.state.apply_fn({'params': self.state.params}, xs)
        mse = np.mean((ys - ys_pred)**2)

        self.info[key_name] = mse


def eval_cases(all_cases, eval_task, key_name='eval_acc', use_mse=False, ignore_err=False):
    try:
        len(eval_task)
    except TypeError:
        eval_task = itertools.repeat(eval_task)

    for c, task in tqdm(zip(all_cases, eval_task), total=len(all_cases)):
        try:
            if use_mse:
                c.eval_mse(task, key_name)
            else:
                c.eval(task, key_name)
        except Exception as e:
            if ignore_err:
                continue
            else:
                raise e


# TODO: fix cost_analysis for FLOPs
def get_flops(fn, *args, **kwargs):
    """Borrowed from flax.nn.tabulate"""
    e = fn.lower(*args, **kwargs).compile()
    cost = e.cost_analysis()
    if cost is None:
        print('warn: unable to estimate flops')
        return 0
    flops = int(cost['flops']) if 'flops' in cost else -1
    return flops
