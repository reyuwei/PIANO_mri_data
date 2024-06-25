####################################################################################
##  Author: Yuwei Li 
##  Email: liyw@shanghaitech.edu.cn
##  PIANO: A Parametric Hand Bone Model from Magnetic Resonance Imaging
##  https://liyuwei.cc/proj/piano
#####################################################################################

from skimage import measure
import SimpleITK as sitk
import numpy as np
import trimesh
import colorsys
from scipy.spatial.transform import Rotation
from scipy.interpolate import Rbf, RegularGridInterpolator

def naive_seg(vol, radius=3):
    """
    Apply segmentation to volume with OtsuThreshold with opening

    Parameters
    ----------
    vol : SimpleITK.Image 

    Returns
    ----------
    mask : SimpltITK.Image 
    threslod : int 
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask = otsu_filter.Execute(vol)

    vectorRadius=(radius,radius,radius)
    kernel=sitk.sitkBall
    fg_mask = sitk.BinaryMorphologicalClosing(mask,vectorRadius,kernel)
    return fg_mask

def generate_seg_mesh(mask_vol, label=1):
    """
    Create mesh from mri volume.

    Parameters
    ----------
    mask_vol : SimpleITK.Image
    label: label id

    Returns
    ----------
    meshes : list of trimesh.mesh
    """

    mask_nda = sitk.GetArrayFromImage(mask_vol).transpose(2,1,0)
    spacing = np.array(mask_vol.GetSpacing())
    verts, faces, normals, values = measure.marching_cubes(mask_nda==label, spacing=spacing)
    faces = np.flip(faces, axis=-1)
    mesh = trimesh.Trimesh(vertices=verts,
                        faces=faces,
                        normals=normals,
                        process=True,
                        validate=True)
    t = np.array(mask_vol.GetOrigin()).reshape(3, 1)
    vd = np.array(mask_vol.GetDirection()).reshape(3, 3)
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = vd
    trans_mat[:3, -1:] = t
    mesh.apply_transform(trans_mat)
    mesh = mesh.simplify_quadratic_decimation(12000)        
    return mesh


def handjoints2mesh(hand_joints, sample=10, use_cylinder=False):
    joint_count = hand_joints.shape[0]
    bone_ske = []
    bone_ske_weight = []

    bone_count = 20
    BONE_PARENT_LABEL_DICT = {
        0: [-1, -1],
        1: [0, 0],
        2: [1, 1],
        3: [2, 2],
        4: [3, 3],
        5: [0, 0],
        6: [5, 4],
        7: [6, 5],
        8: [7, 6],
        9: [8, 7],
        10: [0, 0],
        11: [10, 8],
        12: [11, 9],
        13: [12, 10],
        14: [13, 11],
        15: [0, 0],
        16: [15, 12],
        17: [16, 13],
        18: [17, 14],
        19: [18, 15],
        20: [0, 0],
        21: [20, 16],
        22: [21, 17],
        23: [22, 18],
        24: [23, 19],
    }

    for i in range(1, hand_joints.shape[0]):
        t = np.linspace(0.15, 0.85, sample).reshape(-1, 1)

        joint_parent, joint_weight_id = BONE_PARENT_LABEL_DICT[i]
        one_bone_line = hand_joints[i] + t * (hand_joints[joint_parent] - hand_joints[i])

        # add cylinder
        if use_cylinder:
            bone_length = np.linalg.norm(hand_joints[joint_parent] - hand_joints[i])
            cylinder = trimesh.primitives.Cylinder(height=bone_length * 9. / 10., radius=bone_length / 20.0, sections=8)
            rot_target = (hand_joints[joint_parent] - hand_joints[i]) / bone_length
            rot_from = cylinder.direction
            rot_mat = np.eye(4)
            rot_mat[:3, :3] = Rotation.align_vectors(rot_target[None, ...], rot_from[None, ...])[0].as_matrix()
            cylinder.apply_transform(rot_mat)
            cylinder.apply_translation((hand_joints[i] + hand_joints[joint_parent]) / 2)
            bone_ske.append(np.vstack([cylinder.vertices, np.array(one_bone_line)]))
            weights = np.zeros([t.shape[0] + cylinder.vertices.shape[0], bone_count])
        else:
            bone_ske.append(one_bone_line)
            weights = np.zeros([t.shape[0], bone_count])

        weights[:, joint_weight_id] = 1.0
        bone_ske_weight.append(weights)

    bone_ske = np.stack(bone_ske).reshape(-1, 3)
    bone_ske_weight = np.stack(bone_ske_weight).reshape(-1, bone_count)
    return bone_ske, bone_ske_weight


def RBF_weights(bone_obj, ctrl_pts, weight):

    xyz = bone_obj.vertices.reshape(-1, 3)
    chunk = 50000
    rbfi = Rbf(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2], weight, function="thin_plate", mode="N-D")
    weight_volume = np.concatenate([rbfi(xyz[j:j + chunk, 0], xyz[j:j + chunk, 1], xyz[j:j + chunk, 2]) for j in range(0, xyz.shape[0], chunk)], 0)
    weight_volume[weight_volume < 0] = 0
    weight_volume = weight_volume / np.sum(weight_volume, axis=1).reshape(-1, 1)
    weight_volume = weight_volume.reshape(xyz.shape[0], -1)

    bone_pts_weights = weight_volume
    label_list = np.argmax(bone_pts_weights, axis=1)

    if isinstance(bone_obj.visual, trimesh.visual.TextureVisuals):
        bone_obj.visual = bone_obj.visual.to_color()
        bone_obj.visual.vertex_colors = np.zeros([len(bone_obj.vertices), 4])

    bone_num = bone_pts_weights.shape[-1]
    for i in range(len(label_list)):
        (r, g, b) = colorsys.hsv_to_rgb(label_list[i] * 1.0 / bone_num, 0.8, 0.8)
        bone_obj.visual.vertex_colors[i] = (r * 255, g * 255, b * 255, 255)
        
    return bone_obj


def finegrained_bone(joints3d, bonemesh):
    # generate skeleton
    skeleton_verts, skeleton_weight = handjoints2mesh(joints3d, sample=15, use_cylinder=True)
    semantic_bonemesh = RBF_weights(bonemesh, skeleton_verts, skeleton_weight)
    return semantic_bonemesh


if __name__ == "__main__":

    ## Generate mesh from volume mask
    mri_mask = "00001_bonemuscle.nii"
    mri_mask_vol = sitk.ReadImage(mri_mask)
    bone_mesh = generate_seg_mesh(mri_mask_vol, 1)
    muscle_mesh = generate_seg_mesh(mri_mask_vol, 2)

    bone_mesh.export("bone.obj")
    muscle_mesh.export("muscle.obj")

    ## Naive fine-grained bone mask
    joints_file = "00001_joints.txt"
    joints3d = np.loadtxt(joints_file)
    semantic_bonemesh = finegrained_bone(joints3d, bone_mesh)
    semantic_bonemesh.export("sbone.obj")

    ## Automatic surface segmentation
    mri_raw = "00001.nii"
    surf_mask_vol = naive_seg(sitk.ReadImage(mri_raw))
    surf_mesh = generate_seg_mesh(surf_mask_vol)
    surf_mesh.export("surf.obj")
