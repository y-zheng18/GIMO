python eval.py --save_path chkpoints/gaze3d_crossmodal_mlp --save_fre 1 \
 --val_fre 1 --seq_decay_ratio 1 --gaze_points 10 --batch_size 8 --sample_points 300000 \
 --motion_n_layers 6 --gaze_n_layers 6 --gaze_latent_dim 256 --cross_hidden_dim 256 \
 --cross_n_layers 6 --num_workers 8 --output_path results/gaze3d_demo \
 --load_model_dir chkpoints/crossmodal_pretrained.pth --vposer_path vposer_v1_0 --smplx_path smplx_models --dataroot /orion/u/yangzheng/code/gaze_dataset
