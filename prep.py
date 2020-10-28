import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
import numpy as np

import torch

dataroot = Path('/mnt/datasets/NIPA/comp_2020')

test_df = pd.read_csv(dataroot/'test/test.tsv', header=None)
test_df.columns = ['fn']
test_df['label'] = -1
test_df.head()

train_df = pd.read_csv(dataroot/'train/train.tsv', sep='\t', header=None)
train_df.columns = ['fn', 'l0', 'l1']
train_df['label'] = train_df.groupby(['l0','l1']).ngroup()
num_classes = train_df.label.nunique()
print("num_classes=", num_classes)

label_map = train_df.groupby('label')[['l0','l1']].first().reset_index(drop=False)

SEED=2002
labels = train_df.label.values
pl.trainer.seed_everything(SEED)
skf = StratifiedKFold(shuffle=True, random_state=SEED)
inds = []
for fold_i, (train_ind, val_ind) in enumerate(skf.split(np.zeros_like(labels), labels)):
    train_df.loc[val_ind, 'fold'] = fold_i
train_df.fold = train_df.fold.astype('int')

torch.save({'tr': train_df, 'te': test_df, 'label_map': label_map, 'num_classes': num_classes}, 'data.pt')
