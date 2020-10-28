from settings import dataroot
import torch.nn.functional as F
import torch

infn="submission_val_f1.pt"
subfn=infn.split('.')[0] + ".csv"
submission = torch.load(infn)

o = torch.load('data.pt')
test_df = o['te']
label_map = o['label_map']

submission_ensembled = 0                                          

tot = sum(len(m) for m in submission)                                        

for m in submission:                                            
    for sub in m:
        submission_ensembled += F.softmax(sub, dim=-1) / tot

label = submission_ensembled.argmax(dim=-1)
test_df.label = label
sub_df = test_df.reset_index(drop=False).merge(label_map, on=['label'], how='inner', sort=False).sort_values('index').drop(['index','label'], axis=1).reset_index(drop=True)
sub_df.to_csv(subfn, sep='\t', header=None, index=None)

