from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

dataroot = Path('/mnt/datasets/NIPA/comp_2020')

npyroot = dataroot/'train/npy'
npyroot.mkdir(exist_ok=True)
for p in tqdm((dataroot/'train').glob("*.jpg")):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    fn = p.stem + '.npy'
    np.save(npyroot/fn, img)

npyroot = dataroot/'test/npy'
npyroot.mkdir(exist_ok=True)
for p in tqdm((dataroot/'test').glob("*.jpg")):
    img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    fn = p.stem + '.npy'
    np.save(npyroot/fn, img)
