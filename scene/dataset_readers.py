#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import math
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud


from scene.torf_datasets.tof_dataset import ToFDataset
from scene.torf_datasets.ios_dataset import IOSDataset
from scene.torf_datasets.real_dataset import RealDataset
from scene.torf_datasets.mitsuba_dataset import MitsubaDataset
from scene.torf_datasets.my_utils.my_utils import depth_from_tof
from scene.torf_datasets.my_utils.projection_utils import *


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    znear: Optional[float] = 0.01 # Default value from 3DGS
    zfar: Optional[float] = 100.0
    depth_range: Optional[float] = 100.0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readToRFCameras(dataset, frame_ids, args):
    cam_infos = []
    color_extrinsics = np.array(dataset.dataset["color_extrinsics"])

    for frame_id in frame_ids:
        view_id = dataset.get_view_id[frame_id]
        color_image_path, color_image = "", None
        R, T = np.identity(3).astype(np.float32), np.zeros(3).astype(np.float32)
        R = np.transpose(
            color_extrinsics[view_id, :3, :3]
        )  # torf extrinsics is already w2c
        T = color_extrinsics[view_id, :3, 3]
        color_image_path = dataset._get_color_filename(frame_id)
        color_image = Image.fromarray(
            np.array(dataset.dataset["color_images"][view_id] * 255.0, dtype=np.byte),
            "RGB",
        )

        image_name = dataset._get_image_name(frame_id)

        FovY = 2 * np.arctan2(240, 2 * dataset.tof_intrinsics[view_id][1, 1])  # radian
        FovX = 2 * np.arctan2(320, 2 * dataset.tof_intrinsics[view_id][0, 0])

        cam_infos.append(
            CameraInfo(
                uid=view_id,  # Unique ID of this camera
                R=R,
                T=T,  # RGB camera pose
                image=color_image,
                image_path=color_image_path,  # RGB GT
                FovY=FovY,
                FovX=FovX,
                width=320,
                height=240,  # We assume that the size of color and tof images are the same
                image_name=image_name,
                znear=dataset.dataset['bounds'][0].item(),
                zfar=dataset.dataset['bounds'][1].item()
            )
        )
    return cam_infos


def cameraFrustumCorners(cam_info: CameraInfo):
    """
    Calculate the world-space positions of the corners of the camera's view frustum.
    """
    aspect_ratio = cam_info.width / cam_info.height
    hnear = 2 * np.tan(cam_info.FovY / 2) * cam_info.znear
    wnear = hnear * aspect_ratio
    hfar = 2 * np.tan(cam_info.FovX / 2) * cam_info.zfar
    wfar = hfar * aspect_ratio

    # Camera's forward direction (z forward y down in SfM convention)
    forward = cam_info.R[2]
    right = cam_info.R[0]
    up = -cam_info.R[1]

    # Camera position
    cam_pos = cam_info.T

    # Near plane corners
    nc_tl = cam_pos + forward * cam_info.znear + up * (hnear / 2) - right * (wnear / 2)
    nc_tr = cam_pos + forward * cam_info.znear + up * (hnear / 2) + right * (wnear / 2)
    nc_bl = cam_pos + forward * cam_info.znear - up * (hnear / 2) - right * (wnear / 2)
    nc_br = cam_pos + forward * cam_info.znear - up * (hnear / 2) + right * (wnear / 2)

    # Far plane corners
    fc_tl = cam_pos + forward * cam_info.zfar + up * (hfar / 2) - right * (wfar / 2)
    fc_tr = cam_pos + forward * cam_info.zfar + up * (hfar / 2) + right * (wfar / 2)
    fc_bl = cam_pos + forward * cam_info.zfar - up * (hfar / 2) - right * (wfar / 2)
    fc_br = cam_pos + forward * cam_info.zfar - up * (hfar / 2) + right * (wfar / 2)

    return np.array([nc_tl, nc_tr, nc_bl, nc_br, fc_tl, fc_tr, fc_bl, fc_br])


def calculateSceneBounds(train_cam_infos, args):
    cam_xyzs = np.array([cam_info.T for cam_info in train_cam_infos])
    cam_dirs = np.array(
        [cam_info.R[2] for cam_info in train_cam_infos]
    )  # SfM convention

    all_corners = []
    for cam_info in train_cam_infos:
        corners = cameraFrustumCorners(cam_info)
        all_corners.append(corners)

    # if args.debug:
    #     plt.ioff()
    #     # Visualize camera positions
    #     fig = plt.figure(figsize=(10, 7))
    #     ax = plt.axes(projection="3d")
    #     ax.scatter3D(cam_xyzs[:, 0], cam_xyzs[:, 1], cam_xyzs[:, 2], color="green")

    #     # Visualize camera viewing directions
    #     for i in range(cam_dirs.shape[0]):
    #         view_dir = cam_dirs[i]
    #         scale = 0.05
    #         ax.quiver(
    #             cam_xyzs[i, 0],
    #             cam_xyzs[i, 1],
    #             cam_xyzs[i, 2],
    #             view_dir[0] * scale,
    #             view_dir[1] * scale,
    #             view_dir[2] * scale,
    #             color="red",
    #             length=1,
    #             normalize=True,
    #         )

    #     # Visualize camera corners (to determine scene bounds)
    #     for cs in all_corners:
    #         ax.scatter3D(cs[:, 0], cs[:, 1], cs[:, 2], color="blue")
    #     plt.title("Camera Poses")
    #     plt.legend()
    #     plt.savefig(os.path.join(args.model_path, "scene_bounds.png"))
    #     plt.close()

    all_corners = np.vstack(all_corners)
    min_bounds = np.min(all_corners, axis=0)
    max_bounds = np.max(all_corners, axis=0)

    return min_bounds, max_bounds


def readToFPhasorStaticInfo(path, args, all_args):

    dataset = RealDataset(all_args)
    # elif args.dataset_type == "ios":
    #     dataset = IOSDataset(all_args)
    # elif args.dataset_type == "mitsuba":
    #     dataset = MitsubaDataset(all_args)
    # else:
    #     dataset = ToFDataset(all_args)

    print("Reading training camera info from ToRF")
    print(dataset.i_train)
    train_cam_infos = readToRFCameras(dataset, dataset.i_train, all_args)
    print("Reading testing camera info from ToRF")
    test_cam_infos = readToRFCameras(dataset, dataset.i_test, all_args)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    # if nerf_normalization["radius"] == 0.0:  # camera is fixed
    #     if args.use_tof:
    #         max_depth = np.max(
    #             depth_from_tof(
    #                 train_cam_infos[0].tof_image, args.depth_range, args.phase_offset
    #             )
    #         )
    #         nerf_normalization["radius"] = max_depth * 1.1
    #     else:
    #         print(
    #             "If there is only one view, we must use the ToF image to get the scene radius",
    #             file=sys.stderr,
    #         )
    #         sys.exit(1)

    ply_path = os.path.join(path, "points3d.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)

    colors = None

    num_pts = 50000
    print(f"Generating random point cloud ({num_pts})...")

    # Init xyz
    min_bounds, max_bounds = calculateSceneBounds(train_cam_infos, all_args)
    xyz = np.random.uniform(min_bounds, max_bounds, (num_pts, 3))

    shs_color = RGB2SH(np.ones((num_pts, 3)) * 0.5)
    colors = SH2RGB(shs_color)

    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))

    colors *= 255.0
    storePly(ply_path, xyz, colors)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "ToF": readToFPhasorStaticInfo,
}
