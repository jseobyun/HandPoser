import os
import numpy as np
import cv2
import json
from glob import glob
import os.path as osp

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import smplx
import torch


# mano layer
smplx_path = '/home/jseob/Desktop/yjs/data/mano_v1_2'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True),
              'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:, 0, :] *= -1


def vis_mano(pose, shape=None):
    if np.shape(pose)[0] == 16:
        root_pose = torch.FloatTensor(pose[0]).view(1,3)
        hand_pose = torch.FloatTensor(pose[1:, :]).view(1, -1)
    elif np.shape(pose)[0] == 15:
        root_pose = torch.FloatTensor(np.array([np.pi/2, 0, 0])).view(1, 3)
        hand_pose = torch.FloatTensor(pose).view(1, -1)
    if shape is None:
        shape = np.zeros([1, 10], np.float32)
    shape = torch.FloatTensor(shape).view(1, -1)
    trans = torch.FloatTensor(np.array([0.05, 0.1, 1.0], dtype=np.float32)).view(1, 3)

    output = mano_layer['right'](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
    mesh = output.vertices[0].numpy() * 1000  # meter to milimeter

    t = np.zeros([3, 1])
    R = np.eye(3)

    mesh = np.dot(R, mesh.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)

    # mesh
    mesh = mesh / 1000  # milimeter to meter
    mesh = trimesh.Trimesh(mesh, mano_layer['right'].faces)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # add camera intrinsics
    focal = np.array([1500, 1500], dtype=np.float32).reshape(2)
    princpt = np.array([100, 100], dtype=np.float32).reshape(2)
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=500, viewport_height=500, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    depth = depth[:, :, None]
    valid_mask = (depth > 0)

    render_mask = valid_mask
    img = rgb * render_mask  # + img * (1 - render_mask)
    img = img.astype(np.uint8)

    return img
