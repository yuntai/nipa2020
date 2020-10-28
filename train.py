import pandas as pd
import gc
import pretrainedmodels
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
import os
from lrs_scheduler import WarmRestart

from settings import dataroot

from albumentations import (
  Compose,                
  GaussianBlur,           
  HorizontalFlip,         
  MedianBlur,             
  MotionBlur,             
  Normalize,              
  OneOf,                  
  RandomBrightness,       
  RandomContrast,         
  Resize,                 
  ShiftScaleRotate,       
  VerticalFlip,           
)          

def generate_transforms(image_size):                                                                       
    train_transform = Compose([                                                                                                  
        #Resize(height=image_size[0], width=image_size[1]),                                             
        OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),                     
        OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3)], p=0.5),
        VerticalFlip(p=0.5),                                                                           
        HorizontalFlip(p=0.5),                                                                         
        ShiftScaleRotate(                                                                              
            shift_limit=0.2,                                                                           
            scale_limit=0.2,                                                                           
            rotate_limit=20,                                                                           
            interpolation=cv2.INTER_LINEAR,                                                            
            border_mode=cv2.BORDER_REFLECT_101,                                                        
            p=1,                                                                                       
        ), 
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])                                                                                                      
                                                                                                          
    val_transform = Compose([                                                                                                  
        #Resize(height=image_size[0], width=image_size[1]),                                             
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
    ])                                                                                                      

    return {"train": train_transform, "val": val_transform}                          

class PlantDataset(Dataset):
    def __init__(self, df, transforms, imgroot, num_classes, do_soft_label=False, _lambda=0.3):
        self.df = df
        self.transforms = transforms
        self.imgroot = imgroot
        self.onehot = torch.eye(num_classes)
        self._lambda = _lambda
        self.do_soft_label = do_soft_label

    def __getitem__(self, ind):
        fn = Path(self.df.loc[ind, 'fn']).stem + '.npy'
        img = np.load(str(self.imgroot/fn))
        img = self.transforms(image=img)["image"].transpose(2, 0, 1)
        
        label = torch.tensor(self.df.loc[ind, 'label'])
        onehot = self.onehot[label]
        if self.do_soft_label:
            soft_labels = torch.tensor(self.df.loc[ind, 'soft_label0':'soft_label19'].astype('float').values)
            onehot = onehot * (1.-self._lambda) + soft_labels * self._lambda
        return img, label, onehot

    def __len__(self):
        return len(self.df)


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))

class LitClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        backbone = pretrainedmodels.__dict__["se_resnext50_32x4d"]()
        nets = list(backbone.children())[:-2] + [nn.AvgPool2d(kernel_size=8, stride=1, padding=0), nn.Flatten(), nn.Linear(in_features=2048, out_features=num_classes, bias=True)]
        self.model = nn.Sequential(*nets)
        self.criterion = CrossEntropyLossOneHot()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(opt, T_max=10, T_mult=1, eta_min=1e-5)
        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, y, onehot = batch
        y_hat = self.model(imgs)
        loss = self.criterion(y_hat, onehot)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, y, onehot = batch
        y_hat = self.model(imgs)
        loss = self.criterion(y_hat, onehot)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': y_hat, 'labels': y}

    def validation_epoch_end(self, outputs):
        labels = torch.cat([o['labels'] for o in outputs], dim=0)
        preds = torch.cat([o['preds'] for o in outputs], dim=0)
        f1 = f1_score(labels.cpu().numpy(), preds.argmax(dim=-1).cpu().numpy(), average='micro')
        self.log('val_f1', f1, prog_bar=True, logger=True)

if __name__ == '__main__':

    o = torch.load('data.pt')
    train_df = o['tr']
    num_classes = o['num_classes']
    batch_size = 64
    
    transforms = generate_transforms((256, 256))

    ckptdir='ckpts_noval'
    for fold_i in sorted(train_df.fold.unique()):
        tr_df = train_df[train_df.fold != fold_i].reset_index(drop=True)
        va_df = train_df[train_df.fold == fold_i].reset_index(drop=True)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=6,
            mode="min",
            #filepath=os.path.join(ckptdir, f"fold={fold_i}"+"-{epoch}-{val_loss:.4f}-{val_f1:.4f}"),
            filepath=os.path.join(ckptdir, f"fold={fold_i}"+"-{epoch}-{loss:.4f}"),
            verbose=True,
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
        train_dataset = PlantDataset(tr_df, transforms['train'], dataroot/'train/npy', num_classes, do_soft_label=True)
        val_dataset = PlantDataset(va_df, transforms['val'], dataroot/'train/npy', num_classes, do_soft_label=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

        model = LitClassifier(num_classes)

        trainer = pl.Trainer(
            min_epochs=10,
            max_epochs=20,
            callbacks=[early_stop_callback],
            checkpoint_callback=checkpoint_callback,
            #progress_bar_refresh_rate=0,
            #tpu_cores=8,
            gpus=-1,
            precision=16,
            num_sanity_val_steps=0,
            profiler=False,
            weights_summary=None,
            gradient_clip_val=1,
            distributed_backend='ddp',
            #fast_dev_run=True,
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        del model
        gc.collect()
        torch.cuda.empty_cache()
