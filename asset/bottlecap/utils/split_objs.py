import os
import pickle

import trimesh
import numpy as np
import shutil
from decompose import export_urdf
from scipy.spatial.transform import Rotation as R

new_dir = '../urdf/mesh'
old_dir = '/home/mani/liuqingtao/tv_mani_isaac/asset/bottlecap/xml/bottle_cap/mesh'


def split_obj():
    for filename in os.listdir(old_dir):
        if 'piece_0.obj' in filename:
            obj_name = filename.split('-coacd_convex_piece')[0]
            new_path = os.path.join(new_dir, obj_name)
            if not os.path.exists(new_path):
                os.makedirs(new_path, exist_ok=True)

    for filename in os.listdir(old_dir):
        obj_meshfile_old = os.path.join(old_dir, filename)
        new_path = os.path.join(new_dir, filename.split('-coacd_convex_piece')[0])
        shutil.copy(obj_meshfile_old, new_path)


def get_cap_index(new_dir):

    obj_init_height = dict()
    hand_init_height = dict()

    for dir in os.listdir(new_dir):
        path = os.path.join(new_dir, dir)
        # get all decomposed meshes
        pieces = [piece for piece in os.listdir(path) if 'coacd_convex_piece' in piece]
        if len(pieces) <= 1:
            continue
        peices_mesh = [trimesh.load_mesh(os.path.join(path, piece)) for piece in pieces]
        # # rotate
        # trans = np.eye(4)
        # trans[:-1, :-1] = R.from_euler('x', 90, degrees=True).as_matrix()
        # for id, mesh in enumerate(peices_mesh):
        #     mesh.apply_transform(trans)
        #     mesh.export(os.path.join('../assets', tp + '-' + pieces[id]))
        peices_mesh_z_max = [np.max(mesh.vertices[:, -1]) for mesh in peices_mesh]
        peices_mesh_z_min = [np.min(mesh.vertices[:, -1]) for mesh in peices_mesh]

        # get top piece
        top_piece_name, _ = os.path.splitext(pieces[np.argmax(peices_mesh_z_max)])
        # get min height
        min_height = np.min(peices_mesh_z_min)
        # add top pieces
        obj_init_height[dir] = -min_height / 10
        hand_init_height[dir] = np.max(peices_mesh_z_max - min_height) / 10 + 0.05

    pickle.dump(obj_init_height, open('../urdf/obj_info/obj_init_height.pickle', 'wb'))
    pickle.dump(hand_init_height, open('../urdf/obj_info/hand_init_height.pickle', 'wb'))

    return None


if __name__ == "__main__":
    # split_obj()
    # get_cap_index(new_dir)
    for obj in os.listdir(new_dir):
        obj_dir = os.path.join(new_dir, obj)
        export_urdf(obj_dir, obj_dir)
    print('h')
