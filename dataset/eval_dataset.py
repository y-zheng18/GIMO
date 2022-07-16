import torch.utils.data as data
import torch
from torchvision import transforms
import numpy as np
import random
import os
import json
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation


class EgoEvalDataset(data.Dataset):
    def __init__(self, config, train=False):

        self.config = config
        self.train = train
        self.dataroot = config.dataroot
        self.input_seq_len = config.input_seq_len
        self.output_seq_len = config.output_seq_len
        self.fps = config.fps
        self.disable_gaze = config.disable_gaze

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))
        # print(self.dataset_info)
        self.parse_data_info()
        self.load_scene()

        self.random_ori_list = [-180, -90, 0, 90]
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        ego_idx = self.poses_path_list[index]
        scene = self.scenes_path_list[index]
        seq = self.sequences_path_list[index]
        start_frame, end_frame = self.start_end_list[index]

        imgs = []
        poses_input = []
        poses_input_idx = []
        gazes = []
        gazes_mask = []
        poses_input = []
        smplx_vertices = []
        random_ori = np.random.choice(
            self.random_ori_list)  # np.random.uniform(-self.config.random_angle, self.config.random_angle)
        random_rotation = Rotation.from_euler('xyz', [0, random_ori, 0], degrees=True).as_matrix()

        transform_path = self.trans_path_list[index]
        transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
        scale = transform_info['scale']
        trans_pose2scene = np.array(transform_info['transformation'])
        trans_pose2scene[:3, 3] /= scale
        transform_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape(
            (4, 4))
        transform_norm[:3, 3] /= scale
        # trans_scene2pose = np.linalg.inv(trans_pose2scene)
        transform_pose = transform_norm @ trans_pose2scene

        for f in range(self.input_seq_len):
            pose_idx = ego_idx + int(f * 30 / self.fps)
            poses_input_idx.append(pose_idx)
            # img_data = Image.open(os.path.join(self.dataroot, scene, seq, 'egocentric_imgs',
            #                                    '{}.jpg'.format(str(pose_idx).zfill(4)))).convert('RGB')

            # img_data = self.transform(img_data)

            gaze_points = np.zeros((1, 3))  # (self.config.gaze_points, 3))
            gazes_mask.append(torch.zeros(1).long())

            if not self.disable_gaze:
                gaze_ply_path = os.path.join(self.dataroot, scene, seq, 'eye_pc',
                                             '{}_center.ply'.format(pose_idx))
                gaze_pc_path = os.path.join(self.dataroot, scene, seq, 'eye_pc',
                                            '{}.ply'.format(pose_idx))
                if os.path.exists(gaze_pc_path):

                    gaze_data = trimesh.load_mesh(gaze_ply_path)
                    gaze_data.apply_scale(1 / scale)
                    gaze_data.apply_transform(transform_norm)

                    points = gaze_data.vertices
                    if np.sum(abs(points)) > 1e-8:
                        gazes_mask[-1] = torch.ones(1).long()
                    gaze_points = gaze_data.vertices[
                                  0:1]  # np.random.choice(range(len(points)), self.config.gaze_points)]
                    gaze_pc_data = trimesh.load_mesh(gaze_pc_path)
                    if len(gaze_pc_data.vertices) == 0 or np.sum(abs(gaze_pc_data.vertices)) < 1e-8:
                        gazes_mask[-1] = torch.ones(0).long()
            # imgs.append(img_data)
            pose_data = pickle.load(open(os.path.join(self.dataroot, scene, seq, 'smplx_local',
                                                      '{}.pkl'.format(pose_idx)), 'rb'))
            ori = pose_data['orient'].detach().cpu().numpy()
            trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
            R = Rotation.from_rotvec(ori).as_matrix()

            R_s = transform_pose[:3, :3] @ R
            ori_s = Rotation.from_matrix(R_s).as_rotvec()
            trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

            # poses_input.append(torch.cat([pose_data['orient'], pose_data['trans'], pose_data['latent']]))
            if self.train:
                ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec()
                trans_s = (random_rotation @ trans_s.reshape((3, 1))).reshape(3)

                gaze_points = (random_rotation @ gaze_points.T).T

            poses_input.append(
                torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                           pose_data['latent']]))
            gazes.append(torch.from_numpy(gaze_points).float())
            smplx = trimesh.load_mesh(os.path.join(self.dataroot, scene, seq, 'smplx_local',
                                                   '{}.obj'.format(pose_idx)))

            smplx_vertices.append(torch.from_numpy(smplx.vertices).float())

        # imgs = torch.stack(imgs, dim=0)
        # print(imgs.shape)
        gazes = torch.stack(gazes, dim=0)
        poses_input = torch.stack(poses_input, dim=0).detach()
        gazes_mask = torch.stack(gazes_mask, dim=0)

        gazes_valid_id = torch.where(gazes_mask)
        gazes_invalid_id = torch.where(torch.abs(gazes_mask - 1))
        gazes_valid = gazes[gazes_valid_id]
        gazes[gazes_invalid_id] *= 0
        gazes[gazes_invalid_id] += torch.mean(gazes_valid, dim=0, keepdim=True)
        smplx_vertices = torch.stack(smplx_vertices, dim=0)
        # print(poses_input.shape)

        mask = []
        poses_label = []
        poses_predict_idx = []
        for f in range(self.output_seq_len + 1):
            pose_idx = ego_idx + int(self.input_seq_len * 30 / self.fps) + int(f * 30 / self.fps)
            poses_predict_idx.append(pose_idx)
            pose_path = os.path.join(self.dataroot, scene, seq, 'smplx_local',
                                     '{}.pkl'.format(pose_idx if f < self.output_seq_len else end_frame))

            if not os.path.exists(pose_path) or pose_idx >= end_frame:
                poses_label.append(poses_label[-1])
                mask.append(torch.zeros(1).float())

            else:
                pose_data = pickle.load(open(pose_path, 'rb'))

                ori = pose_data['orient'].detach().cpu().numpy()
                trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
                R = Rotation.from_rotvec(ori).as_matrix()
                R_s = transform_pose[:3, :3] @ R
                ori_s = Rotation.from_matrix(R_s).as_rotvec()
                trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)


                # poses_label.append(torch.cat([pose_data['orient'], pose_data['trans'], pose_data['latent']]))
                if self.train:
                    ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec()
                    trans_s = (random_rotation @ trans_s.reshape((3, 1))).reshape(3)
                poses_label.append(
                    torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                               pose_data['latent']]))
                mask.append(torch.ones(1).float())
        poses_label = torch.stack(poses_label, dim=0).detach()
        poses_mask = torch.cat(mask)

        scene_points = self.scene_list['{}_{}'.format(seq, start_frame)]
        scene_points = scene_points[np.random.choice(range(len(scene_points)), self.config.sample_points)]
        scene_points *= 1 / scale
        scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T
        if self.train:
            scene_points = (random_rotation @ scene_points.T).T
            scene_points += np.random.normal(loc=0, scale=self.config.sigma, size=scene_points.shape)

        return gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, torch.from_numpy(scene_points).float(), seq, scene, poses_predict_idx, poses_input_idx

    def __len__(self):
        return len(self.poses_path_list)

    def parse_data_info(self):
        self.sequences_path_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            start_frame = self.dataset_info['start_frame'][i]
            end_frame = self.dataset_info['end_frame'][i]
            scene = self.dataset_info['scene'][i]
            transform = self.dataset_info['transformation'][i]
            # print(start_frame, end_frame, seq)
            self.poses_path_list.append(start_frame)
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(scene)
            self.trans_path_list.append(transform)
            self.start_end_list.append([self.dataset_info['start_frame'][i], self.dataset_info['end_frame'][i]])

    def load_scene(self):
        self.scene_list = {}
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            print('loading scene of {}'.format(seq))
            scene = self.dataset_info['scene'][i]
            transform_path = self.dataset_info['transformation'][i]
            start_frame = self.dataset_info['start_frame'][i]
            transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
            scale = transform_info['scale']
            trans_pose2scene = np.array(transform_info['transformation'])
            trans_scene2pose = np.linalg.inv(trans_pose2scene)

            scene_ply = trimesh.load_mesh(os.path.join(self.dataroot, scene, 'scene_obj', 'scene_downsampled.ply'))
            # print(scene_ply.vertices.shape)

            scene_points = scene_ply.vertices
            # scene_points = (trans_scene2pose[:3, :3] @ scene_points.T + trans_scene2pose[:3, 3:]).T
            # scene_ply.apply_transform(trans_scene2pose)
            # scene_ply.apply_scale(1 / scale)
            # scene_points *= 1 / scale
            # points = scene_ply.vertices
            # points = scene_points[np.random.choice(range(len(scene_points)), self.config.sample_points, replace=False)]

            self.scene_list['{}_{}'.format(seq, start_frame)] = scene_points



