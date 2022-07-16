import torch
import numpy as np
from dataset import ego_dataset, eval_dataset
import time
import torch.nn as nn
from human_body_prior.tools.model_loader import load_vposer
import smplx
from config.config import MotionFromGazeConfig
from model.multimodal_net import multimodal_net
import trimesh
from tqdm import tqdm
import os


class save_smplx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

        self.body_mesh_model = smplx.create(config.smplx_path,
                                       model_type='smplx',
                                       gender='neutral', ext='npz',
                                       num_pca_comps=12,
                                       create_global_orient=True,
                                       create_body_pose=True,
                                       create_betas=True,
                                       create_left_hand_pose=True,
                                       create_right_hand_pose=True,
                                       create_expression=True,
                                       create_jaw_pose=True,
                                       create_leye_pose=True,
                                       create_reye_pose=True,
                                       create_transl=True,
                                       batch_size=1,
                                       num_betas=10,
                                       num_expression_coeffs=10)

    def forward(self, poses_input, poses_label, poses_predict, gazes, scene_points, prefix='epoch0'):
        save_path = os.path.join(self.config.save_path, '{}_smplx'.format(prefix))
        os.makedirs(save_path, exist_ok=True)
        with torch.no_grad():
            for i, p in enumerate(poses_input[0]):
                pose = {}
                body_pose = self.vposer.decode(p[6:], output_type='aa')
                pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                pose['transl'] = p[3:6].cpu().unsqueeze(0)
                smplx_output = self.body_mesh_model(return_verts=True,
                                               **pose)
                body_verts_batch = smplx_output.vertices
                smplx_faces = self.body_mesh_model.faces
                out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                out_mesh.export(os.path.join(save_path, 'input_{}.obj'.format(i)))

                gaze_ply = trimesh.PointCloud(gazes[0, i].cpu().numpy(), colors=np.ones((gazes[0].shape[1], 3)))
                gaze_ply.export(os.path.join(save_path, 'input_{}_gaze.ply').format(i))
            for i, p in enumerate(poses_label[0]):
                pose = {}
                body_pose = self.vposer.decode(p[6:], output_type='aa')
                pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                pose['transl'] = p[3:6].cpu().unsqueeze(0)
                smplx_output = self.body_mesh_model(return_verts=True,
                                               **pose)
                body_verts_batch = smplx_output.vertices
                smplx_faces = self.body_mesh_model.faces
                out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                out_mesh.export(os.path.join(save_path, 'gt_{}.obj'.format(i)))
            for i, p in enumerate(poses_predict[0]):
                pose = {}
                body_pose = self.vposer.decode(p[6:], output_type='aa')
                pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                pose['transl'] = p[3:6].cpu().unsqueeze(0)
                smplx_output = self.body_mesh_model(return_verts=True,
                                               **pose)
                body_verts_batch = smplx_output.vertices
                smplx_faces = self.body_mesh_model.faces
                out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                out_mesh.export(os.path.join(save_path, 'predict_{}.obj'.format(i)))
            scene_ply = trimesh.PointCloud(scene_points[0].cpu().numpy(), colors=np.ones((scene_points.shape[1], 3)))
            scene_ply.export(os.path.join(save_path, 'scene.ply'))