"""Ben's ensemble stuff"""

# <codecell>
from dataclasses import dataclass, field
import itertools

import pandas as pd

import sys
sys.path.append('../../../../')
from common import *
from train import *
from model.cnn import CnnConfig, EnsembleConfig
from task.vision import Vision 
from task.vision.data_stats import CLASS_DICT


@dataclass
class Case:
    name: str
    config: dataclass
    train_args: dict = field(default_factory=dict)
    state: list = None
    hist: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def run(self, train_task, test_task, **args):
        self.state, hist = train(self.config, loss='ce_onehot', data_iter=train_task, test_iter=test_task, **self.train_args, **args)
        self.hist.append(hist)
    
    def eval(self, task, key_name='eval_acc'):
        all_accs = []
        for xs, ys in task:
            logits = self.state.apply_fn({'params': self.state.params}, xs)
            if len(logits.shape) == 3:
                logits = jnp.mean(logits, axis=0)

            preds = logits.argmax(axis=1)

            if len(ys.shape) > 1:
                ys = ys.argmax(axis=1)

            eval_acc = np.mean(ys == preds)
            all_accs.append(eval_acc)

        self.info[key_name] = np.mean(all_accs)
    
    def eval_mse(self, task, key_name='eval_mse'):
        xs, ys = next(task)
        ys_pred = self.state.apply_fn({'params': self.state.params}, xs)
        mse = np.mean((ys - ys_pred)**2)

        self.info[key_name] = mse

run_id = new_seed()

run_split = 5

n_out = CLASS_DICT['cifar10']
n_total_epochs = 100

bs = [0]
# ls = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
ls = 10**np.linspace(-5, -3, num=6)
total_params = [351_700]
ks = np.arange(1, 11)
gs = 10**np.linspace(-2, 2, num=11)

def _get_c(n_params):
    num = -133 + np.sqrt(133**2 - 4 * 690 * (10 - n_params))
    den = 2 * 690
    return np.round(num / den).astype(int)

### START TEST CONFIGS
# run_split = 1

# n_total_epochs = 2
# ls = [1e-5]
# total_params = [10_000]
# ks = [1]
# gs = [1]
### END TEST CONFIGS

def get_config(c, freeze=False):
    return CnnConfig(
        cnn_widths=[c, 2 * c],
        mlp_widths=[5 * c],
        freeze_intermediate_layers=freeze,
        n_out=n_out
    )

learned_cases = []

for b, l, n_params, k, g in itertools.product(bs, ls, total_params, ks, gs):
    n_mem_params = n_params // k
    c_mem = _get_c(n_mem_params)

    if g <= 1:
        lr = g**2 * 0.1
    else:
        lr = g * 0.1

    full_gamma = g * np.sqrt(5 * c_mem)

    learned_cases.append(
        Case(f'Ensemble (k={k})', 
             EnsembleConfig(member_config=get_config(c_mem, freeze=False)),
             train_args={'n_epochs': n_total_epochs, 'test_every': 5, 'beta': b, 'weight_decay': l, 'lr': lr, 'early_stop_n': None, 'optim': sgd_wd, 'gamma': full_gamma},
             info={'k': k, 'n_params': n_params, 'g': g})
    )

train_task = Vision('cifar10', batch_size=128, data_resolution=32, augment=False)
test_task = Vision('cifar10', batch_size=128, data_resolution=32, mode='test', augment=False)

all_cases = split_cases(learned_cases, run_split=run_split)
print('CASES', all_cases)

for case in (all_cases):
    case.run(iter(train_task), iter(test_task), use_freeze=False)

for case in all_cases:
    eval_task = Vision(name='cifar10', batch_size=1024, data_resolution=32, mode='test', augment=False)
    case.eval(eval_task)
    case.state = None

    # Vision tasks are not picklable
    # case.train_task = None
    # case.test_task = None

df = pd.DataFrame(all_cases)
df.to_pickle(f'res.{run_id}.pkl')

print('done!')


# %%
