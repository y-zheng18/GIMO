import numpy as np
import json
import os
import math
import argparse

import bpy


def read_obj_file(obj_file_path):
    '''
    Load .obj file, return vertices, faces.
    return: vertices: N_v X 3, faces: N_f X 3
    '''
    obj_f = open(obj_file_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == 'f':  # Need to consider / case, // case, etc.
            faces.append([int(line[3].split('/')[0]),
                          int(line[2].split('/')[0]),
                          int(line[1].split('/')[0]) \
                          ])  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
    obj_f.close()

    return np.asarray(vertices), np.asarray(faces)


if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment for HuMoR Generation.')
    parser.add_argument('--folder', type=str, metavar='PATH',
                        help='path to specific folder which include folders containing .obj files',
                        default='')
    parser.add_argument('--scene', type=str, metavar='PATH',
                        help='path to specific .ply path for 3D scene',
                        default='')
    parser.add_argument('--output_folder', type=str,
                        help='path to save imgs',
                        default='')
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    ## Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)

    ori_obj_folders = os.listdir(args.folder)
    obj_folders = [o for o in ori_obj_folders if '.obj' in o]
    # obj_folders = []
    # for folder_name in ori_obj_folders:
    #     if "blender_out" not in folder_name and "N0SittingBooth_00169_01" in folder_name:
    #         obj_folders.append(folder_name)
    # obj_folders.sort()
    #
    # scene_name = args.scene.split("/")[-1].replace("_scene.blend", "")
    # print("scene name:{0}".format(scene_name))
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    for obj_folder_name in obj_folders:
        # if "blender_out" not in obj_folder_name and scene_name in obj_folder_name:
        path_to_file = os.path.join(args.folder, obj_folder_name)
        # output_dir = os.path.join(args.folder, obj_folder_name + "_blender_out")
        obj_id = obj_folder_name[:-4]

        print("obj_file:{0}".format(path_to_file))
        print("output dir:{0}".format(os.path.join(output_folder, ("%04d" % int(obj_id)) + ".png")))

        # Iterate folder to process all model
        # path_to_file = os.path.join(obj_folder, file_name)
        new_obj = bpy.ops.import_scene.obj(filepath=path_to_file, split_mode="OFF")
        # obj_object = bpy.context.selected_objects[0]
        obj_object = bpy.data.objects[obj_id]
        # obj_object.scale = (0.3, 0.3, 0.3)
        mesh = obj_object.data
        for f in mesh.polygons:
            f.use_smooth = True

        # obj_object.rotation_euler = (math.radians(0), math.radians(0), math.radians(
        #     0))  # The default seems 90, 0, 0 while importing .obj into blender
        # obj_object.location.y = 0

        mat = bpy.data.materials.new(name="MaterialName")  # set new material to variable
        obj_object.data.materials.append(mat)
        mat.use_nodes = True
        principled_bsdf = mat.node_tree.nodes['Principled BSDF']
        if principled_bsdf is not None:
            # principled_bsdf.inputs[0].default_value = (153/255.0, 51/255.0, 255/255.0, 1)
            # principled_bsdf.inputs[0].default_value = (26/255.0, 26/255.0, 26/255.0, 1)
            principled_bsdf.inputs[0].default_value = (220 / 255.0, 220 / 255.0, 220 / 255.0, 1)
            # principled_bsdf.inputs[0].default_value = (10/255.0, 0/255.0, 255/255.0, 1)

        obj_object.active_material = mat

        # bpy.data.scenes['Scene'].render.filepath = os.path.join(output_dir, file_name.replace(".obj", ".png"))
        bpy.data.scenes['Scene'].render.filepath = os.path.join(output_folder, ("%04d" % int(obj_id)) + ".png")
        bpy.ops.render.render(write_still=True)

        # Delet materials
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        bpy.data.objects.remove(obj_object, do_unlink=True)

