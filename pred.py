from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from settings import dataroot
from train import LitClassifier, PlantDataset, generate_transforms

def parse_meta(fn):
    return dict(x.split('=') for x in fn.split('-'))

def pick_best_ckpt(ckptdir):
    n_splits = 5
    ens_ckpts = []
    for fold_i in range(n_splits):
        ckpts = list(Path(ckptdir).glob(f"fold={fold_i}*"))
        metas = {c: parse_meta(c.stem) for c in ckpts}
        ens_ckpts.append(max(metas.keys(), key=lambda k: float(metas[k]['val_f1'])))
        #ens_ckpts.append(min(metas.keys(), key=lambda k: float(metas[k][metric])))
    ens_ckpts.sort(key=lambda x: int(parse_meta(x.stem)['fold']))
    return ens_ckpts        

ckpts = pick_best_ckpt("ckpts_long_train")
for c in ckpts:
    print(str(c))

o = torch.load('data.pt')
num_classes =  o['num_classes']
test_df = o['te']
tr = o['tr']

model = LitClassifier(num_classes)
transforms = generate_transforms((256, 256))

if 1:
    tta_ensemble_cnt = 8
    test_dataset = PlantDataset(test_df, transforms['train'], dataroot/'test/npy', num_classes)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    submission = []

    for ckpt in ckpts:
        print(f"`running {ckpt}`")
        model.load_state_dict(torch.load(ckpt)["state_dict"])
        model.cuda()
        model.eval()

        tta_preds = []
        for _ in range(tta_ensemble_cnt):
            preds = []
            with torch.no_grad():
                for image, *_ in test_dataloader:
                    preds.append(model(image.cuda()).cpu())
                tta_preds.append(torch.cat(preds))

        submission.append(tta_preds)
    torch.save(submission, 'submission_val_f1.pt')

if 0:
    print("building soft label...")

    tr[[f'soft_label{i}' for i in range(num_classes)]] = -1

    for fold_i in tr.fold.unique():
        print(f"{fold_i=}")
        va_df = tr[tr.fold == fold_i].reset_index(drop=True) # allow using .loc inside PlantDataset
        val_dataset = PlantDataset(va_df, transforms=transforms["val"], imgroot=dataroot/'train/npy')
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        model.load_state_dict(torch.load(ckpts[fold_i])["state_dict"])
        model.to("cuda")
        model.eval()

        preds = []
        with torch.no_grad():
            for image, label in tqdm(val_dataloader):
                preds.append(model(image.to("cuda")).cpu())
            preds = torch.cat(preds)
        preds = F.softmax(preds, dim=-1)
        tr.loc[tr.fold == fold_i, 'soft_label0':'soft_label19'] = preds
    assert (tr.loc[:, 'soft_label0':'soft_label19'] == -1).sum().sum()
    torch.save(train_df, 'train_long_train.pt')
