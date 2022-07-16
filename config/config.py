from argparse import ArgumentParser


class MotionFromGazeConfig(ArgumentParser):
    def __init__(self):
        super().__init__()

        self.input_configs = self.add_argument_group('input')
        self.input_configs.add_argument('--batch_size', default=2, type=int)
        self.input_configs.add_argument('--num_workers', default=4, type=int)
        self.input_configs.add_argument('--input_seq_len', default=6, type=int)
        self.input_configs.add_argument('--output_seq_len', default=10, type=int)
        self.input_configs.add_argument('--seq_decay_ratio', default=0.8, type=float)
        self.input_configs.add_argument('--dataroot', default='./demo_data', type=str)
        self.input_configs.add_argument('--fps', default=2, type=int)
        self.input_configs.add_argument('--disable_gaze', default=False, action='store_true')
        self.input_configs.add_argument('--disable_scene', default=False, action='store_true')
        self.input_configs.add_argument('--disable_crossmodal', default=False, action='store_true')
        self.input_configs.add_argument('--scene_feats_dim', default=256, type=int)
        self.input_configs.add_argument('--image_feats_dim', default=256, type=int)
        self.input_configs.add_argument('--motion_dim', default=38, type=int)
        self.input_configs.add_argument('--sample_points', default=200000, type=int)
        self.input_configs.add_argument('--gaze_points', default=1500, type=int)
        self.input_configs.add_argument('--random_angle', default=30, type=float)
        self.input_configs.add_argument('--random_trans', default=3, type=float)
        self.input_configs.add_argument('--sigma', default=0.01, type=float)

        self.add_argument('--model_type', default='cross', choices=['cross', 'mint', 'rnn', 'st_transformer', 'simple'])
        self.motion_configs = self.add_argument_group('motion_transformer')
        self.motion_configs.add_argument('--dropout', default=0, type=float)
        self.motion_configs.add_argument('--motion_n_heads', default=8, type=int)
        self.motion_configs.add_argument('--motion_hidden_dim', default=256, type=int)
        self.motion_configs.add_argument('--motion_intermediate_dim', default=1024, type=int)
        self.motion_configs.add_argument('--motion_n_layers', default=3, type=int)
        self.motion_configs.add_argument('--motion_latent_dim', default=256, type=int)

        self.gaze_configs = self.add_argument_group('gaze_transformer')
        self.gaze_configs.add_argument('--gaze_n_heads', default=8, type=int)
        self.gaze_configs.add_argument('--gaze_hidden_dim', default=256, type=int)
        self.gaze_configs.add_argument('--gaze_intermediate_dim', default=1024, type=int)
        self.gaze_configs.add_argument('--gaze_n_layers', default=3, type=int)
        self.gaze_configs.add_argument('--gaze_latent_dim', default=256, type=int)

        self.cross_modal_configs = self.add_argument_group('cross_modal_transformer')
        self.cross_modal_configs.add_argument('--cross_n_heads', default=8, type=int)
        self.cross_modal_configs.add_argument('--cross_hidden_dim', default=256, type=int)
        self.cross_modal_configs.add_argument('--cross_intermediate_dim', default=1024, type=int)
        self.cross_modal_configs.add_argument('--cross_n_layers', default=3, type=int)

        self.train_configs = self.add_argument_group('train')
        self.train_configs.add_argument('--pointnet_chkpoints', default='pretrained/point.model', type=str)

        self.train_configs.add_argument('--save_path', type=str, default='chkpoints/')
        self.train_configs.add_argument('--save_fre', type=int, default=5)
        self.train_configs.add_argument('--vis_fre', type=int, default=1000)
        self.train_configs.add_argument('--val_fre', type=int, default=3)
        self.train_configs.add_argument('--load_model_dir', type=str, default=None)
        self.train_configs.add_argument('--load_optim_dir', type=str, default=None)

        self.train_configs.add_argument('--epoch', type=int, default=50)
        self.train_configs.add_argument('--lr', type=float, default=1e-4)
        self.train_configs.add_argument('--weight_decay', type=float, default=1e-4)
        self.train_configs.add_argument('--gamma', type=float, default=0.99)
        self.train_configs.add_argument('--lambda_ori', type=float, default=1)
        self.train_configs.add_argument('--lambda_trans', type=float, default=1)
        self.train_configs.add_argument('--lambda_latent', type=float, default=1)
        self.train_configs.add_argument('--lambda_rec', type=float, default=1)
        self.train_configs.add_argument('--lambda_des', type=float, default=1)

        self.eval_configs = self.add_argument_group('eval')
        self.eval_configs.add_argument('--output_path', default='results', type=str)
        self.eval_configs.add_argument('--smplx_path', default='smplx_models', type=str)
        self.eval_configs.add_argument('--vposer_path', default='vposer_v1_0', type=str)

    def get_configs(self):
        return self.parse_args()


if __name__ == '__main__':
    config = MotionFromGazeConfig()
    print(config.get_configs())
