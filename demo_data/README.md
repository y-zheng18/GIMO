## Data structure

The dataset is structured as follows: 

```
dataset.csv				# annotation information
scene_xxx/
    scene_obj/
        textured_output.obj		# scanned mesh of the scene
        scene_downsampled.ply   # downsampled scene
        transform_norm.txt      # to normailze the scene into the canonical space
    sequence_name/
        xxx_head_hand_eye.csv	# collected head, hand, eye gaze pose 
        xxx_pv.txt				# camera poses of egocentric images
        eye_pc/
            xxx.ply	            # gaze point cloud
        PV/xxx.png				# egocentric images
        smplx_local/
            xxx.obj			# smplx mesh
            xxx.pkl			# latent embedding of smplx, using VPoser to decode
        transform_infox.json		# transform smplx to the 3d scene
        transform_scene2scene.json	# transform poses of egocentric images to the 3d scene
        skeleton.bvh                # the original collected motion in bvh format

```
Please refer to [eval_dataset.py](../dataset/eval_dataset.py) for details of transforming motions, gaze and the 3d scenes into the same coordinate space.


