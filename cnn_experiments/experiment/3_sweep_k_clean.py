"""Ben's ensembling project"""
# <codecell>
import itertools
import jax

import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import from_state_dict
import optax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.optimize import minimize


import sys
sys.path.append('../')

from common import *
from train import train, create_train_state
from model.cnn import CnnConfig, EnsembleConfig, ResNetConfig

from task.vision import Vision

set_theme()

# <codecell>
df = collate_dfs('remote/3_sweep_k_clean/scan_k_no_aug')
df

# <codecell>
def extract_plot_vals(row):
    return pd.Series([
        row['name'],
        row['info']['k'],
        np.log10(row['info']['g']),
        np.log10(row['train_args']['weight_decay']),
        row['info']['eval_acc'].item()
    ], index=['name', 'k', 'log10_gamma', 'log10_lambda', 'eval_acc'])

plot_df = df.apply(extract_plot_vals, axis=1)
plot_df

# <codecell>
mdf = plot_df.drop('name', axis=1)
mean_accs = mdf.groupby(['log10_lambda', 'log10_gamma', 'k'], as_index=True).mean()

ks = np.unique(plot_df['k'])
gs = np.unique(plot_df['log10_gamma'])
ls = np.unique(plot_df['log10_lambda'])

optim_lambdas = []
optim_accs = []

for k, g in itertools.product(ks, gs):
    accs = [mean_accs['eval_acc'][l, g.item(), k.item()].item() for l in ls]
    max_idx = np.argmax(accs)
    res = {'log10_lambda_optim': ls[max_idx], 'k': k, 'log10_gamma': g}
    optim_lambdas.append(res)
    optim_accs.append(mdf[(mdf['log10_gamma'] == g) & (mdf['k'] == k) & (mdf['log10_lambda'] == ls[max_idx])])


optim_accs = pd.concat(optim_accs)
# <codecell>
# optim_accs = optim_accs[optim_accs['log10_gamma'] < 0.6]

optim_accs = optim_accs[
    (optim_accs['log10_gamma'] != -1.8) &
    (optim_accs['log10_gamma'] != -1.4) &
    (optim_accs['log10_gamma'] != -1.0) &
    (optim_accs['log10_gamma'] != -0.6) &
    (optim_accs['log10_gamma'] != -0.2)
]

g = sns.lineplot(optim_accs, x='k', y='eval_acc', hue='log10_gamma', marker='o', legend='auto')
g.set_xlabel('K')
g.set_ylabel('Test Accuracy')

g.legend_.set_title(r'$\gamma$')
for t in g.legend_.texts:
    label = t.get_text()
    t.set_text('${10}^{%s}$' % label)

g.figure.tight_layout()

sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))

plt.savefig('fig/k_sweep_no_aug_no_es.svg', bbox_inches='tight')

# <codecell>
mdf = plot_df
mdf['log10_gamma'] = np.round(mdf['log10_gamma'], decimals=1)

gs = sns.relplot(mdf, 
                 x='k', y='eval_acc', 
                 hue='log10_lambda', 
                 col='log10_gamma', 
                 col_wrap=3, 
                 kind='line', 
                 marker='o',
                 height=2, aspect=1.3)


gs.legend.set_title(r'$\lambda$')
for t in gs.legend.texts:
    label = t.get_text()
    t.set_text('${10}^{%s}$' % label)

gs.set_xlabels('K')
gs.set_ylabels('Test Acc.')

gammas = np.sort(np.unique(mdf['log10_gamma']))
for ax, gamma in zip(gs.axes, gammas):
    ax.set_title(rf'$\gamma = 10^{ {gamma.item()} }$')

gs.figure.savefig('fig/varying_gamma_and_lambda_no_aug_no_es.svg', bbox_inches='tight')

# <codecell>
ldf = pd.DataFrame(optim_lambdas)
g = sns.lineplot(ldf, x='k', y='log10_lambda_optim', hue='log10_gamma', marker='o')

sns.move_legend(g, 'upper left', bbox_to_anchor=(1, 1))
g.figure.savefig('fig/optim_lambdas.png')

# <codecell>
sns.lineplot(plot_df, x='k', y='eval_acc', hue='gamma', marker='o', legend='full')
# plt.savefig('fig/k_sweep.png')


# <codecell>
### TRAINING PLAYGROUND

def _get_c(n_params):
    num = -133 + np.sqrt(133**2 - 4 * 690 * (10 - n_params))
    den = 2 * 690
    return np.round(num / den).astype(int)

_get_c(351700)

# <codecell>

# config = CnnConfig(n_out=10,
#                    cnn_widths=[23, 46],
#                    mlp_widths=[115],
#                    freeze_intermediate_layers=False)

config = ResNetConfig(n_out=100,
                   stage_sizes=[2, 2, 2, 2],
                   n_features=64,
                   freeze_intermediate_layers=False)

ens_config = EnsembleConfig(config, n_members=1)
# ens_config = CnnConfig(
#     cnn_widths=[50, 100],
#     mlp_widths=[400],
#     n_out=10
# )

# <codecell>
task = Vision('cifar100', batch_size=128, data_resolution=32)
test_task = Vision('cifar100', batch_size=128, data_resolution=32, mode='test', augment=False)

state, hist = train(ens_config, 
                    data_iter=iter(task), 
                    test_iter=iter(test_task), 
                    loss='ce_onehot', 
                    test_every=1, 
                    n_epochs=50, 
                    beta=0, 
                    use_freeze=False, 
                    optim=optax.sgd,
                    # lr=0.1,
                    lr=optax.schedules.cosine_decay_schedule(
                        init_value=10 * 0.1,
                        decay_steps=40_000,
                        alpha=1e-3
                    ),
                    momentum=0.9,
                    gamma=10 * np.sqrt(64), 
                    # optim=optax.sgd, lr=10 * 0.1,
                    # momentum=0.9
                    )



# <codecell>
# Jeffares result plotting
path = r'/home/grandpaa/workspace/ensemble/jeffares/results/notag_CIFAR-10_CNN_20240924_15:39:33/results.csv'
df = pd.read_csv(path)

sns.lineplot(df, x='beta', y='test_acc_ens', marker='o')
plt.savefig('fig/learner_collusion_code_vanilla_adam.png')

# %%
