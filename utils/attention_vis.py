import torch
import numpy as np
from dataset import ego_dataset, eval_dataset
import time
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer
import smplx
from config.config import MotionFromGazeConfig
from model.spatial_temporal_transformer import ST_net
from model.crossmodal_net import crossmodal_net
from model.motion_gaze_net import motion_gaze_net
from model.RNN import motion_rnn
import trimesh
from tqdm import tqdm
import os
import json
import cv2
from einops import repeat
import torch.nn as nn
np.random.seed(42)


class Attention_vis():
    def __init__(self, config):
        self.config = config
        self.test_dataset = eval_dataset.EgoEvalDataset(config, train=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def eval(self, model=None):
        if model is None:
            model = crossmodal_net(config)

            model = model.to(self.device)
            assert self.config.load_model_dir is not None
            print('loading pretrained model from ', self.config.load_model_dir)
            model.load_state_dict(torch.load(self.config.load_model_dir))
            print('load done!')
        with torch.no_grad():
            model.eval()

            for data in tqdm(self.test_dataset):

                gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points, seq, scene, poses_predict_idx, poses_input_idx = data
                if '{}_{}'.format(seq, poses_input_idx[0]) != self.config.vis_seq:
                    continue
                gazes = gazes.unsqueeze(0).to(self.device)
                gazes_mask = gazes_mask.unsqueeze(0).to(self.device)
                poses_input = poses_input.unsqueeze(0).to(self.device)
                smplx_vertices = smplx_vertices.unsqueeze(0).to(self.device)
                poses_label = poses_label.unsqueeze(0).to(self.device)
                scene_points = scene_points.unsqueeze(0).to(self.device).contiguous()
                # print(gazes.shape, scene_points.shape)
                poses_predict = model(gazes, gazes_mask, poses_input, smplx_vertices, scene_points)

                bs, seq_len, _, _ = gazes.shape
                n_points = scene_points.shape[1]
                scene_feats, scene_global_feats = model.scene_encoder(scene_points.repeat(1, 1, 2))  # B x F x M, B x F

                motion_feats = model.motion_linear(poses_input)

                motion_scene_feats = model.fp_layer(smplx_vertices.reshape((bs * seq_len, -1, 3)),
                                                    scene_points.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape(
                                                       (bs * seq_len, -1, 3)),
                                                   scene_feats.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape(
                                                       (bs * seq_len, -1, n_points)))
                # B*seq_len x F x N_smpl
                motion_scene_feats = model.pointnet(motion_scene_feats).reshape((bs, seq_len, -1))

                motion_feats = torch.cat([motion_feats, motion_scene_feats], dim=2)
                motion_embedding = model.motion_encoder(motion_feats)
                out_seq_len = motion_embedding.shape[1]
                cross_modal_embedding = scene_global_feats.unsqueeze(1).repeat(1, out_seq_len, 1)

                gazes, _ = torch.median(gazes, dim=2)  # B x seq_len x 3

                gaze_embedding = model.fp_layer(gazes, scene_points, scene_feats).permute((0, 2, 1))  # B x seq_len x F
                gaze_embedding = model.gaze_linear(gaze_embedding)
                # gaze_embedding *= gazes_mask
                gaze_embedding = model.gaze_encoder(gaze_embedding)
                gaze_motion_embedding, motion_att = model.gaze_motion_decoder(gaze_embedding, motion_embedding, True)
                motion_gaze_embedding, gaze_att = model.motion_gaze_decoder(motion_embedding, gaze_embedding, True)

                gaze_att_img = self.draw(gaze_att[0, :, :-1] * 2)
                motion_att_img = self.draw(motion_att[0, :, :-1] * 2, 'blue')
                os.makedirs(self.config.output_path, exist_ok=True)
                cv2.imwrite(os.path.join(self.config.output_path, 'gaze_att_input.jpg'), gaze_att_img)
                cv2.imwrite(os.path.join(self.config.output_path, 'motion_att_input.jpg'), motion_att_img)
                return motion_att, gaze_att

    def draw(self, atts, color='red'):
        x = np.ones((1900, 1900, 3)) * 255
        img = x.astype(np.uint8)
        for i in range(atts.shape[0]):
            for j in range(atts.shape[1]):
                img = cv2.circle(img, (100 + j * 100, 100 + i * 100), radius=30, 
                                 color=[0, 0, 255 * atts[i, j].cpu().numpy()] if color=='red' else [255 * atts[i, j].cpu().numpy(), 0, 0], thickness=-1)
        return img

if __name__ == '__main__':
    config = MotionFromGazeConfig()
    config.add_argument('--vis_seq', default='2022-02-21-014246_1660', type=str)
    config = config.parse_args()
    att = Attention_vis(config).eval()