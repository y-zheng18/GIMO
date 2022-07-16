import torch
import numpy as np
from dataset import eval_dataset
import time
import torch.nn.functional as F
from human_body_prior.tools.model_loader import load_vposer
import smplx
from config.config import MotionFromGazeConfig
from model.crossmodal_net import crossmodal_net
import trimesh
from tqdm import tqdm
import os
import json

np.random.seed(42)


class SMPLX_evalutor():
    def __init__(self, config):
        self.config = config
        self.test_dataset = eval_dataset.EgoEvalDataset(config, train=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vposer, _ = load_vposer(self.config.vposer_path, vp_model='snapshot')
        self.vposer = self.vposer.to(self.device)

    def eval(self, model=None):
        if model is None:
            if config.model_type == 'cross':
                model = crossmodal_net(config)
            else:
                raise NotImplementedError
            model = model.to(self.device)
            assert self.config.load_model_dir is not None
            print('loading pretrained model from ', self.config.load_model_dir)
            model.load_state_dict(torch.load(self.config.load_model_dir))
            print('load done!')
        with torch.no_grad():
            model.eval()

            body_mesh_model = smplx.create(self.config.smplx_path,
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
            loss_dict = {}
            for data in tqdm(self.test_dataset):

                gazes, gazes_mask, poses_input, smplx_vertices, poses_label, poses_mask, scene_points, seq, scene, poses_predict_idx, poses_input_idx = data

                # imgs = imgs.unsqueeze(0).to(device)
                gazes = gazes.unsqueeze(0).to(self.device)
                gazes_mask = gazes_mask.unsqueeze(0).to(self.device)
                poses_mask = poses_mask.unsqueeze(0).to(self.device)
                poses_input = poses_input.unsqueeze(0).to(self.device)
                smplx_vertices = smplx_vertices.unsqueeze(0).to(self.device)
                poses_label = poses_label.unsqueeze(0).to(self.device)
                scene_points = scene_points.unsqueeze(0).to(self.device).contiguous()
                # print(gazes.shape, scene_points.shape)
                poses_predict = model(gazes, gazes_mask, poses_input, smplx_vertices, scene_points)
                # print(poses_predict.shape)
                save_path = os.path.join(self.config.output_path, '{}_{}_{}'.format(scene, seq, poses_input_idx[0]))

                loss_des_ori = F.l1_loss(poses_predict[:, -1, :3], poses_label[:, -1, :3])
                loss_des_trans = F.l1_loss(poses_predict[:, -1, 3:6], poses_label[:, -1, 3:6])
                loss_des_latent = F.l1_loss(poses_predict[:, -1, 6:], poses_label[:, -1, 6:])

                loss_all = F.l1_loss(poses_predict[:, self.config.input_seq_len:-1, :], poses_label[:, :-1],
                                     reduction='none')
                # print(poses_mask)
                loss_rec = F.l1_loss(poses_predict[:, :self.config.input_seq_len, :], poses_input)
                loss_all *= poses_mask[:, :-1].unsqueeze(2)

                loss_ori = (loss_all[:, :, :3].sum(dim=1) / poses_mask.sum(dim=1, keepdim=True)).mean()
                loss_trans = (loss_all[:, :, 3:6].sum(dim=1) / poses_mask.sum(dim=1, keepdim=True)).mean()
                loss_latent = (loss_all[:, :, 6:].sum(dim=1) / poses_mask.sum(dim=1, keepdim=True)).mean()

                loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])] = {'path_trans_error': loss_trans.item(),
                                                                                'path_ori_error': loss_ori.item(),
                                                                                'path_latent_error': loss_latent.item(),
                                                                                'des_trans_error': loss_des_trans.item(),
                                                                                'des_ori_error': loss_des_ori.item(),
                                                                                'des_latent_error': loss_des_latent.item(),
                                                                                'rec_loss': loss_rec.item()}

                os.makedirs(save_path, exist_ok=True)
                for i, p in enumerate(poses_input[0]):
                    pose = {}
                    body_pose = self.vposer.decode(p[6:], output_type='aa')
                    pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                    pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                    pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                    pose['transl'] = p[3:6].cpu().unsqueeze(0)
                    smplx_output = body_mesh_model(return_verts=True,
                                                   **pose)
                    body_verts_batch = smplx_output.vertices
                    smplx_faces = body_mesh_model.faces
                    out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                    out_mesh.export(os.path.join(save_path, 'input_{}.obj'.format(poses_input_idx[i])))

                    gaze_ply = trimesh.PointCloud(gazes[0, i].cpu().numpy(), colors=np.ones((gazes[0].shape[1], 3)))
                    gaze_ply.export(os.path.join(save_path, 'input_{}_gaze.ply').format(poses_input_idx[i]))
                gt_pose = []
                predict_pose = []
                gt_joints = []
                predict_joints = []
                for i, p in enumerate(poses_label[0]):
                    pose = {}
                    body_pose = self.vposer.decode(p[6:], output_type='aa')
                    pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                    pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                    pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                    pose['transl'] = p[3:6].cpu().unsqueeze(0)
                    smplx_output = body_mesh_model(return_verts=True,
                                                   **pose)
                    body_verts_batch = smplx_output.vertices
                    smplx_faces = body_mesh_model.faces
                    out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                    out_mesh.export(os.path.join(save_path, 'gt_{}.obj'.format(poses_predict_idx[i])))
                    gt_pose.append(pose['body_pose'])
                    gt_joints.append(smplx_output.joints[0])
                    
                poses_input_idx.extend(poses_predict_idx)
                for i, p in enumerate(poses_predict[0]):
                    pose = {}
                    body_pose = self.vposer.decode(p[6:], output_type='aa')
                    pose['body_pose'] = body_pose.cpu().unsqueeze(0)
                    pose['pose_embedding'] = p[6:].cpu().unsqueeze(0)
                    pose['global_orient'] = p[:3].cpu().unsqueeze(0)
                    pose['transl'] = p[3:6].cpu().unsqueeze(0)
                    smplx_output = body_mesh_model(return_verts=True,
                                                   **pose)

                    body_verts_batch = smplx_output.vertices
                    smplx_faces = body_mesh_model.faces
                    out_mesh = trimesh.Trimesh(body_verts_batch[0].cpu().numpy(), smplx_faces, process=False)

                    out_mesh.export(os.path.join(save_path, '{}.obj'.format(poses_input_idx[i])))
                    predict_pose.append(pose['body_pose'])
                    predict_joints.append(smplx_output.joints[0])
                # for i in range(len(poses_label) - 1):
                gt_smplx_aligned = [body_mesh_model(return_verts=True,
                                                    **{'body_pose': p, 'global_orient': torch.zeros((1, 3)),
                                                       'transl': torch.zeros((1, 3))}).joints[0] for p in gt_pose]
                predicted_smplx_aligned = [body_mesh_model(return_verts=True,
                                                           **{'body_pose': p, 'global_orient': torch.zeros((1, 3)),
                                                              'transl': torch.zeros((1, 3))}).joints[0] for p in
                                           predict_pose]

                gt_smplx_aligned = torch.stack(gt_smplx_aligned, dim=0)
                predicted_smplx_aligned = torch.stack(predicted_smplx_aligned, dim=0)

                # print(gt_smplx_aligned.shape, gt_smplx_aligned[0, [0, 1, 2, 3, 4], :],  gt_pose[0].shape)

                gt_smplx_aligned -= (gt_smplx_aligned[:, [1], :] + gt_smplx_aligned[:, [2], :]) / 2
                predicted_smplx_aligned -= (predicted_smplx_aligned[:, [1], :] + predicted_smplx_aligned[:, [2], :]) / 2
                gt_smplx = torch.stack(gt_joints, dim=0)
                predicted_smplx = torch.stack(predict_joints, dim=0)
                gt_smplx -= (gt_smplx[:, [1], :] + gt_smplx[:, [2], :]) / 2
                predicted_smplx -= (predicted_smplx[:, [1], :] + predicted_smplx[:, [2], :]) / 2
                # print(gt_smplx.shape)

                path_MPJPE = torch.norm(gt_smplx[:-1, :23] - predicted_smplx[self.config.input_seq_len:-1, :23],
                                        dim=2).mean()
                des_MPJPE = torch.norm(gt_smplx[-1:, :23] - predicted_smplx[-1:, :23], dim=2).mean()
                path_P_MPJPE = torch.norm(
                    gt_smplx_aligned[:-1, :23] - predicted_smplx_aligned[self.config.input_seq_len:-1, :23],
                    dim=2).mean()
                des_P_MPJPE = torch.norm(gt_smplx_aligned[-1:, :23] - predicted_smplx_aligned[-1:, :23], dim=2).mean()

                loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])]['path_P-MPJPE'] = path_P_MPJPE.item()
                loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])]['des_P-MPJPE'] = des_P_MPJPE.item()
                loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])]['path_MPJPE'] = path_MPJPE.item()
                loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])]['des_MPJPE'] = des_MPJPE.item()

                scene_ply = trimesh.PointCloud(scene_points[0].cpu().numpy(),
                                               colors=np.ones((scene_points.shape[1], 3)))
                scene_ply.export(os.path.join(save_path, 'scene.ply'))
                #print('{}_{}_{}'.format(scene, seq, poses_input_idx[0]),
                #      loss_dict['{}_{}_{}'.format(scene, seq, poses_input_idx[0])])
            mean_rec_loss = np.array([loss_dict[k]['rec_loss'] for k in loss_dict.keys()]).mean()
            mean_path_trans_loss = np.array([loss_dict[k]['path_trans_error'] for k in loss_dict.keys()]).mean()
            mean_path_ori_loss = np.array([loss_dict[k]['path_ori_error'] for k in loss_dict.keys()]).mean()
            mean_des_trans_loss = np.array([loss_dict[k]['des_trans_error'] for k in loss_dict.keys()]).mean()
            mean_des_ori_loss = np.array([loss_dict[k]['des_ori_error'] for k in loss_dict.keys()]).mean()
            mean_path_p_mpjpe = np.array([loss_dict[k]['path_P-MPJPE'] for k in loss_dict.keys()]).mean()
            mean_path_mpjpe = np.array([loss_dict[k]['path_MPJPE'] for k in loss_dict.keys()]).mean()
            mean_des_p_mpjpe = np.array([loss_dict[k]['des_P-MPJPE'] for k in loss_dict.keys()]).mean()
            mean_des_mpjpe = np.array([loss_dict[k]['des_MPJPE'] for k in loss_dict.keys()]).mean()
            print('path trans error:{}'.format(mean_path_trans_loss))
            print('path ori error:{}'.format(mean_path_ori_loss))
            print('path P-MPJPE:{}'.format(mean_path_p_mpjpe))
            print('path MPJPE:{}'.format(mean_path_mpjpe))
            print('rec loss:{}'.format(mean_rec_loss))
            print('des trans error:{}'.format(mean_des_trans_loss))
            print('des ori error:{}'.format(mean_des_ori_loss))
            print('des P-MPJPE:{}'.format(mean_des_p_mpjpe))
            print('des MPJPE:{}'.format(mean_des_mpjpe))

            des_trans_train_loss = np.array(
                [loss_dict[k]['des_trans_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            des_ori_train_loss = np.array(
                [loss_dict[k]['des_ori_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            des_p_mpjpe_train_loss = np.array(
                [loss_dict[k]['des_P-MPJPE'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            des_mpjpe_train_loss = np.array(
                [loss_dict[k]['des_MPJPE'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            des_latent_train_loss = np.array(
                [loss_dict[k]['des_latent_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            des_trans_test_loss = np.array(
                [loss_dict[k]['des_trans_error'] for k in loss_dict.keys() if '_0221' in k]).mean()
            des_ori_test_loss = np.array(
                [loss_dict[k]['des_ori_error'] for k in loss_dict.keys() if '_0221' in k]).mean()
            des_p_mpjpe_test_loss = np.array(
                [loss_dict[k]['des_P-MPJPE'] for k in loss_dict.keys() if '_0221' in k]).mean()
            des_mpjpe_test_loss = np.array(
                [loss_dict[k]['des_MPJPE'] for k in loss_dict.keys() if '_0221' in k]).mean()
            des_latent_test_loss = np.array(
                [loss_dict[k]['des_latent_error'] for k in loss_dict.keys() if '_0221' in k]).mean()

            path_trans_train_loss = np.array(
                [loss_dict[k]['path_trans_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            path_ori_train_loss = np.array(
                [loss_dict[k]['path_ori_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            path_p_mpjpe_train_loss = np.array(
                [loss_dict[k]['path_P-MPJPE'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            path_mpjpe_train_loss = np.array(
                [loss_dict[k]['path_MPJPE'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            path_latent_train_loss = np.array(
                [loss_dict[k]['path_latent_error'] for k in loss_dict.keys() if '_0221' not in k]).mean()
            path_trans_test_loss = np.array(
                [loss_dict[k]['path_trans_error'] for k in loss_dict.keys() if '_0221' in k]).mean()
            path_ori_test_loss = np.array(
                [loss_dict[k]['path_ori_error'] for k in loss_dict.keys() if '_0221' in k]).mean()
            path_p_mpjpe_test_loss = np.array(
                [loss_dict[k]['path_P-MPJPE'] for k in loss_dict.keys() if '_0221' in k]).mean()
            path_mpjpe_test_loss = np.array(
                [loss_dict[k]['path_MPJPE'] for k in loss_dict.keys() if '_0221' in k]).mean()
            path_latent_test_loss = np.array(
                [loss_dict[k]['path_latent_error'] for k in loss_dict.keys() if '_0221' in k]).mean()

            print('training scenes')
            print('path trans error:{}'.format(path_trans_train_loss))
            print('path ori error:{}'.format(path_ori_train_loss))
            print('path latent error:{}'.format(path_latent_train_loss))
            print('path P-MPJPE:{}'.format(path_p_mpjpe_train_loss))
            print('path MPJPE:{}'.format(path_mpjpe_train_loss))
            print('des trans error:{}'.format(des_trans_train_loss))
            print('des ori error:{}'.format(des_ori_train_loss))
            print('des latent error:{}'.format(des_latent_train_loss))
            print('des P-MPJPE:{}'.format(des_p_mpjpe_train_loss))
            print('des MPJPE:{}'.format(des_mpjpe_train_loss))
            print('new scenes')
            print('path trans error:{}'.format(path_trans_test_loss))
            print('path ori error:{}'.format(path_ori_test_loss))
            print('path latent error:{}'.format(path_latent_test_loss))
            print('path P-MPJPE:{}'.format(path_p_mpjpe_test_loss))
            print('path MPJPE:{}'.format(path_mpjpe_test_loss))
            print('des trans error:{}'.format(des_trans_test_loss))
            print('des ori error:{}'.format(des_ori_test_loss))
            print('des latent error:{}'.format(des_latent_test_loss))
            print('des P-MPJPE:{}'.format(des_p_mpjpe_test_loss))
            print('des MPJPE:{}'.format(des_mpjpe_test_loss))

        return loss_dict, mean_path_trans_loss, mean_path_ori_loss, mean_path_p_mpjpe, mean_des_trans_loss, mean_des_ori_loss, mean_des_p_mpjpe


if __name__ == '__main__':
    config = MotionFromGazeConfig().parse_args()
    start = time.time()
    evaluator = SMPLX_evalutor(config)
    r = evaluator.eval()
    json.dump(r[0], open(os.path.join(config.output_path, 'loss.json'), 'w'), indent=1)
