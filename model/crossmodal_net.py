from model.base_cross_model import *
from model.pointnet_plus2 import *

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import os
from config.config import MotionFromGazeConfig
from copy import deepcopy


class crossmodal_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.use_gaze = not config.disable_gaze
        self.use_crossmodal = not config.disable_crossmodal
        # self.use_scene = not config.disable_scene

        self.fp_layer = MyFPModule()
        self.pointnet = PointNet(config.scene_feats_dim)
        self.scene_encoder = PointNet2SemSegSSGShape({'feat_dim': config.scene_feats_dim})

        self.motion_linear = nn.Linear(config.motion_dim, config.motion_hidden_dim)
        input_len = config.input_seq_len
        output_len = config.output_seq_len + config.input_seq_len + 1 # rec + path predict + destination
        if self.use_crossmodal:
            self.motion_scene_encoder = PerceiveEncoder(n_input_channels=config.scene_feats_dim,
                                                        n_latent=output_len,
                                                        n_latent_channels=config.motion_latent_dim,
                                                        n_self_att_heads=config.motion_n_heads,
                                                        n_self_att_layers=config.motion_n_layers,
                                                        dropout=config.dropout)
            self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim + config.scene_feats_dim,
                                                  n_latent=output_len,
                                                  n_latent_channels=config.motion_latent_dim,
                                                  n_self_att_heads=config.motion_n_heads,
                                                  n_self_att_layers=config.motion_n_layers,
                                                  dropout=config.dropout)
            self.motion_decoder = PerceiveDecoder(n_query_channels=config.motion_latent_dim,
                                                  n_query=output_len,
                                                  n_latent_channels=config.motion_latent_dim,
                                                  dropout=config.dropout)           # (bs, out_len, motion_latent_d)
        else:
            self.motion_encoder = PerceiveEncoder(n_input_channels=config.motion_hidden_dim + config.scene_feats_dim,
                                                  n_latent=output_len,
                                                  n_latent_channels=config.motion_latent_dim,
                                                  n_self_att_heads=config.motion_n_heads,
                                                  n_self_att_layers=config.motion_n_layers,
                                                  dropout=config.dropout)           # (bs, out_len, motion_latent_d)

        # if self.use_gaze:
        self.gaze_linear = nn.Linear(config.scene_feats_dim, config.gaze_hidden_dim)
        self.gaze_encoder = PerceiveEncoder(n_input_channels=config.gaze_hidden_dim,
                                            n_latent=output_len,
                                            n_latent_channels=config.gaze_latent_dim,
                                            n_self_att_heads=config.gaze_n_heads,
                                            n_self_att_layers=config.gaze_n_layers,
                                            dropout=config.dropout)             # (bs, out_len, gaze_latent_d)
        if self.use_crossmodal:
            self.gaze_motion_decoder = PerceiveDecoder(n_query_channels=config.gaze_latent_dim,
                                                       n_query=output_len,
                                                       n_latent_channels=config.motion_latent_dim,
                                                       dropout=config.dropout)  # (bs, out_len, gaze_latent_d)
            self.motion_gaze_decoder = PerceiveDecoder(n_query_channels=config.motion_latent_dim,
                                                       n_query=output_len,
                                                       n_latent_channels=config.gaze_latent_dim,
                                                       dropout=config.dropout)  # (bs, out_len, gaze_latent_d)
                # self.cross_modal_gaze_decoder = PerceiveDecoder(n_query_channels=config.motion_latent_dim,
                #                                            n_query=output_len,
                #                                            n_latent_channels=config.gaze_latent_dim,
                #                                            dropout=config.dropout)  # (bs, out_len, gaze_latent_d)
        embedding_dim = config.scene_feats_dim
        if self.use_gaze and self.use_crossmodal:
            embedding_dim += config.gaze_latent_dim + config.motion_latent_dim
        elif self.use_gaze and not self.use_crossmodal:
            embedding_dim += config.gaze_latent_dim + config.motion_latent_dim
        elif not self.use_gaze and self.use_crossmodal:
            embedding_dim += config.gaze_latent_dim + config.motion_latent_dim # config.motion_latent_dim
        else:
            embedding_dim += config.motion_latent_dim
        print('embedding dim: ', embedding_dim)
        self.embedding_layer = PositionwiseFeedForward(embedding_dim, embedding_dim)
        self.output_encoder = PerceiveEncoder(n_input_channels=embedding_dim,
                                              n_latent=output_len,
                                              n_latent_channels=config.cross_hidden_dim,
                                              n_self_att_heads=config.cross_n_heads,
                                              n_self_att_layers=config.cross_n_layers,
                                              dropout=config.dropout)             # (bs, out_len, gaze_latent_d)
        self.outputlayer = nn.Linear(config.cross_hidden_dim, config.motion_dim)


    def forward(self, gazes, gazes_mask, motions, smplx_vetices, scenes):
        """
        :param gazes: (bs, seq_len, n, 3)
        :param gazes_mask: (bs, seq_len)
        :param motions: (bs, seq_len, motion_dim)
        :param smplx_vetices: (bs, seq_len, smplx_vertex_num)
        :param scene: (bs, num_points, 3)
        :return:
        """
        bs, seq_len, _, _ = gazes.shape
        n_points = scenes.shape[1]
        scene_feats, scene_global_feats = self.scene_encoder(scenes.repeat(1, 1, 2))  # B x F x M, B x F

        motion_feats = self.motion_linear(motions)

        motion_scene_feats = self.fp_layer(smplx_vetices.reshape((bs * seq_len, -1, 3)),
                                           scenes.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape((bs * seq_len, -1, 3)),
                                           scene_feats.unsqueeze(1).repeat(1, seq_len, 1, 1).reshape(
                                               (bs * seq_len, -1, n_points)))
        # B*seq_len x F x N_smpl
        motion_scene_feats = self.pointnet(motion_scene_feats).reshape((bs, seq_len, -1))


        if self.use_crossmodal:
            # motion_scene_feats = self.motion_scene_encoder(motion_scene_feats)
            motion_feats = torch.cat([motion_feats, motion_scene_feats], dim=2)
            motion_embedding = self.motion_encoder(motion_feats)
            # motion_embedding = self.motion_decoder(motion_scene_feats, motion_feats)
        else:
            motion_feats = torch.cat([motion_feats, motion_scene_feats], dim=2)
            motion_embedding = self.motion_encoder(motion_feats)

        out_seq_len = motion_embedding.shape[1]
        cross_modal_embedding = scene_global_feats.unsqueeze(1).repeat(1, out_seq_len, 1)

        if self.use_gaze:
            gazes, _ = torch.median(gazes, dim=2)  # B x seq_len x 3

            gaze_embedding = self.fp_layer(gazes, scenes, scene_feats).permute((0, 2, 1))  # B x seq_len x F
            # print(gaze_embedding.shape)
        else:
            gaze_embedding = torch.zeros((bs, seq_len, self.config.scene_feats_dim)).float().to(self.device)
        gaze_embedding = self.gaze_linear(gaze_embedding)
        # gaze_embedding *= gazes_mask
        gaze_embedding = self.gaze_encoder(gaze_embedding)
        if self.use_crossmodal:
            gaze_motion_embedding = self.gaze_motion_decoder(gaze_embedding, motion_embedding)
            motion_gaze_embedding = self.motion_gaze_decoder(motion_embedding, gaze_embedding)
            #print(motion_gaze_embedding.shape, cross_modal_embedding.shape)
            cross_modal_embedding = torch.cat([cross_modal_embedding, gaze_motion_embedding, motion_gaze_embedding], dim=2)
        else:
            cross_modal_embedding = torch.cat([cross_modal_embedding, motion_embedding, gaze_embedding], dim=2)
        # else:
        #     cross_modal_embedding = torch.cat([cross_modal_embedding, motion_embedding], dim=2)

        cross_modal_embedding = self.embedding_layer(cross_modal_embedding)
        cross_modal_embedding = self.output_encoder(cross_modal_embedding)
        output = self.outputlayer(cross_modal_embedding)

        return output


if __name__ == '__main__':
    configs = MotionFromGazeConfig().get_configs()

    net = crossmodal_net(configs).cuda()
    torch.save(net.state_dict(), 'test.pth')
    gazes = torch.randn((2, 6, 1000, 3)).cuda()
    gazes_mask = torch.randn((2, 6)).cuda()
    motions = torch.randn((2, 6, 38)).cuda()
    smplx_vertices = torch.randn((2, 6, 1234, 3)).cuda()
    scenes = torch.randn((2, 100000, 3)).cuda()

    outputs = net(gazes, gazes_mask, motions, smplx_vertices, scenes)
    print(outputs.shape)