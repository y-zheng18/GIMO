import trimesh
import numpy as np
import os
import argparse
import json
import pandas as pd

args = argparse.ArgumentParser()
args.add_argument('--data_idx', default=37, type=int)
args.add_argument('--dataroot', default='./GIMO_dataset/')


opt = args.parse_args()

anno = pd.read_csv(os.path.join(opt.dataroot, 'dataset.csv'))
data_block = anno.iloc[opt.data_idx]
scene_id = data_block['scene']
start_frame = data_block['start_frame']
end_frame = data_block['end_frame']
transform_p2s = data_block['transformation']
seq = data_block['sequence_path']

smplx_root = '{}/{}/{}/smplx_local'.format(opt.dataroot, scene_id, seq)
output_root = '{}/{}/{}/smplx_aligned_fine'.format(opt.dataroot, scene_id, seq)
transform_save_root = '{}/{}/{}/{}'.format(opt.dataroot, scene_id, seq, transform_p2s)
os.makedirs(output_root, exist_ok=True)

transform_info = json.load(open(transform_save_root, 'r'))
scale = transform_info['scale']
print(transform_info)

for i in range(start_frame, end_frame):
    obj_id = '{}.obj'.format(i)
    obj = trimesh.load(os.path.join(smplx_root, obj_id))

    obj.apply_scale(scale)
    obj.apply_transform(transform_info['transformation'])
    obj.export(os.path.join(output_root, obj_id))
