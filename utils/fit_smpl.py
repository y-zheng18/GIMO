import math

import torch
import numpy as np
import cv2
import smplx
import pickle
import trimesh
import os

# fix conflict between mmpose and pyrender in windows delete mmpose\core\visualization line13 os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
from pyrender.constants import RenderFlags
from sklearn.preprocessing import normalize
from human_body_prior.train.vposer_smpl import VPoser
from smplx.lbs import vertices2joints
from torch.utils.data import Dataset
from pytorch3d.transforms.rotation_conversions import *
import time, random
from scipy.spatial.transform import Rotation

class SmplFitter():
    def __init__(self, config, gender='male'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gender = gender
        self.smpl = smplx.SMPLX(model_path=config.smplx_path,model_type='smplx',
                               gender='neutral', ext='npz',
                                use_pca=True, num_pca_comps=45, flat_hand_mean=True)

    def parse_corr(self, keypoints_results):
        view_size = len(keypoints_results)
        whole_2d_results = np.zeros([view_size, 133, 3])
        for view in range(view_size):
            if not keypoints_results[view]:
                continue
            whole_2d_results[view] = keypoints_results[view][0]['keypoints']

        return {
            'body': whole_2d_results[:, :17],
            'foot': whole_2d_results[:, 17:23],
            'face': whole_2d_results[:, 23:91],
            'lhand': whole_2d_results[:, 91:112],
            'rhand': whole_2d_results[:, 112:133],

        }

    def smpl_forward(self, smpl_param):
        smpl_out = self.smpl(
            transl=smpl_param['trans'].unsqueeze(0),
            global_orient=smpl_param['orient'].unsqueeze(0),
            body_pose=smpl_param['body_pose'].unsqueeze(0),
            jaw_pose=smpl_param['jaw_pose'].unsqueeze(0),
            betas=smpl_param['beta'].unsqueeze(0),
            expression=smpl_param['expression'].unsqueeze(0),
            left_hand_pose=smpl_param['lhand'].unsqueeze(0),
            right_hand_pose=smpl_param['rhand'].unsqueeze(0),

        )
        # print(smpl_param['scale'], smpl_out.joints.shape)

        return smpl_out, smpl_out.joints #* smpl_param['scale']

    def init_RT(self, body_skel):
        smpl_out = self.smpl()
        joints = smpl_out.joints[0].detach().cpu().numpy()
        src = joints[np.array([4, 5, 17, 18])].transpose()
        tar = body_skel[np.array([4, 5, 17, 18])].transpose()

        # solve RT
        mu1, mu2 = src.mean(axis=1, keepdims=True), tar.mean(axis=1, keepdims=True)
        X1, X2 = src - mu1, tar - mu2

        K = X1.dot(X2.T)
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        R = V.dot(Z.dot(U.T))
        t = mu2 - R.dot(mu1)

        orient, _ = cv2.Rodrigues(R)
        return orient.reshape(3), t.reshape(3)

    def init_param(self, skel3d):
        orient, trans = self.init_RT(skel3d)# / scale)

        pca_mean = np.array([
            -1.46245074, 0.16150913, -0.13613739, -1.38515116, 0.25973073, -0.0247116, 0.06849018, 0.44786085,
            0.66518973, 0.72898975, -0.0081996, 0.48195812, 1.12141122, -1.49678825, 0.51509144,
            0.78260062, -1.66513595, -1.41780118, 0.10227966, -0.81155625, 0.83845112, -0.05956686, 1.31395593,
            1.2262315, -2.58328008, -0.42809623, -1.62884562, -0.2638993, 1.57074572, -0.92486433,
            -0.57469231, -0.2910394, 1.40550742, 0.10315603, -0.10990491, 0.49792807, 2.34563255, -1.64109268,
            -2.98774547, -1.54570805, -0.672661, 0.05149607, 0.4546438, -1.60890671, 0.29886834])

        smpl_param = {'orient': orient, 'trans': trans, 'body_pose': np.zeros((21, 3)), 'jaw_pose': np.zeros(3),
                      'beta': np.zeros(10), 'expression': np.zeros(10), 'lhand': pca_mean, 'rhand': pca_mean,}
                      # 'scale': float(scale)}
        return smpl_param

    def calc_3d_loss(self, x1, x2):
        loss = torch.nn.functional.mse_loss(x1, x2)
        return loss

    def gen_closure(self, optimizer, smpl_param, corr, w_body2d=None, w_foot2d=None,
                    w_face2d=None, w_hand_reg=None, w_hand2d=None, w_verts3d=None, w_latent=None,
                    w_beta=None, w_jaw=None, w_expression=None, w_ori=None, head_ori=None):
        def closure():
            optimizer.zero_grad()
            smpl_out, skel = self.smpl_forward(smpl_param)
            loss = {}

            loss['body'] = w_body2d * self.calc_3d_loss(skel[0][:23], corr[:23])

            loss = sum(loss.values())
            loss.backward()
            return loss
        return closure


    def solve(self, smpl_param, closure, optimizer, iter_max=500, iter_thresh=1e-10, loss_thresh=5e-10):
        loss_prev = float('inf')
        optimize_beta = True
        for i in range(iter_max):
            loss = optimizer.step(closure).item()
            if abs(loss-loss_prev) < iter_thresh and loss < loss_thresh:
                print('iter ' + str(i) + ': ' + str(loss))
                break
            else:
                #print('iter ' + str(i) + ': ' + str(loss))
                loss_prev = loss
            if loss < loss_thresh and i >= iter_max:
                break
        return smpl_param

    def export(self, filename, smpl_param):
        smpl_out, _ = self.smpl_forward(smpl_param)
        mesh = trimesh.Trimesh(vertices=smpl_out.vertices[0].detach().cpu().numpy(), faces=self.smpl.faces)
        mesh.export(filename)

        return mesh


    def fit(self, skel_3d):
        smpl_param = self.init_param(skel_3d)

        smpl_param = {k: torch.tensor(v, dtype=torch.float32).requires_grad_(True) for k, v in
                      smpl_param.items()}
        smpl_param['beta'].requires_grad = False
        smpl_param['expression'].requires_grad = False
        smpl_param['jaw_pose'].requires_grad = False
        w_body2d = 1
        w_foot2d = 1e-3
        w_latent = 1e-5
        w_beta = 1e-3
        w_jaw = 1e-4
        w_expression = 1e-4
        w_hand_reg = 1e-5
        w_face_2d = 1e-3
        w_hand_2d = [1e-1]
        w_ori = 0
        w_verts3d = 1e-1
        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)
        smpl_param = self.solve(smpl_param=smpl_param, optimizer=optimizer, closure=self.gen_closure(
            optimizer, smpl_param, torch.from_numpy(skel_3d), w_body2d=w_body2d, w_foot2d=w_foot2d, w_latent=w_latent, w_beta=w_beta,
            w_jaw=w_jaw, w_expression=w_expression, w_face2d=w_face_2d, w_hand2d=w_hand_2d[0],
            w_hand_reg=w_hand_reg,
            w_ori=w_ori), iter_max=500)
        return smpl_param

