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
        xxx_pv.txt	        # camera poses of egocentric images
        eye_pc/
            xxx.ply	        # gaze point cloud
        PV/xxx.png		# egocentric images
        smplx_local/
            xxx.obj			# smplx mesh
            xxx.pkl			# latent embedding of smplx, using VPoser to decode
        transform_infox.json		# transform smplx to the 3d scene
        transform_scene2scene.json	# transform poses of egocentric images to the 3d scene
        skeleton.bvh                # the original collected motion in bvh format

```
Please refer to [eval_dataset.py](../dataset/eval_dataset.py) for details of transforming motions, gaze and the 3d scenes into the same coordinate space.

To render the data as shown in the paper, firstly run the following code to align the 3d scene and the smplx mesh:
```
python -m utils.align_smpl --dataroot /path/to/dataset --data_idx 37
```
where `--data_idx` is the index of the sequence in the annotation file (*dataset.csv*). For example, `--data_idx 37` means the *bedroom0210/2022-02-10-031338* sequence. You might want to look up the annotation file (*dataset.csv*) to find the corresponding index of the sequence you want to render.

The renderer is implemented in [render.py](../utils/render_blender.py), which depends on [Blender](https://www.blender.org/). You need to firstly mannual import the scene into blender, add camera, and then save the .blend file. You should get something like this [render.blend](./bedroom0210/render.blend). 
Then you can run the following code to render the scene:
```
/path/to/blender --python ./utils/render_blender.py -- --folder './GIMO_dataset/bedroom0210/2022-02-10-031338/smplx_aligned_fine' \
 --scene './GIMO_dataset/bedroom0210/render.blend' --output_folder './GIMO_dataset/bedroom0210/rendering'
```
You might need to change the path to blender and the path to the scene file. The rendered images will be saved in the `--output_folder`.
