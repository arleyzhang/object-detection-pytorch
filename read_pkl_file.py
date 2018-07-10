# -*- coding:utf-8 -*-  
import os
import pickle
import numpy as np

pkl_file_root = './coco300_35k/minival'

aps = []
for file in os.listdir(pkl_file_root):
    if file.endswith(".pkl"):
        if 'detections' in file: continue
        pkl_path = os.path.join(pkl_file_root, file)
        inf = pickle.load(open(pkl_path, 'rb'))
        # print(pkl_path, inf['ap'].dtype)
        # print(inf['prec'].shape, inf['rec'].shape, inf['ap'].shape)
        aps += [inf['ap']]
print('Mean AP = {:.4f}'.format(np.mean(aps)))


inf = pickle.load(open('/home/maolei/data/VOCdevkit/VOC2007/annotations_cache/annots.pkl', 'rb'))

print (len(inf))