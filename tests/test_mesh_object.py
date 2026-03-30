from pathlib import Path

import importlib.util
import pytest
import torch

from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.objects import MeshObject


MESH_PATH = Path("/home/haegu/mj_sim/assets/syringe/assets/syringe.obj")
HAS_TORCHSDF = importlib.util.find_spec("torchsdf") is not None


def test_mesh_object_initialization_smoke():
    if not MESH_PATH.exists():
        return
    hand_model = ShadowHandModel.create(device="cpu")
    mesh_object = MeshObject(MESH_PATH)
    grasp_state = initialize_grasps_for_primitive(hand_model, mesh_object, batch_size=2, num_contacts=4)
    assert grasp_state.joint_values.shape == (2, 24)
    assert grasp_state.wrist_translation.shape == (2, 3)
    assert grasp_state.wrist_rotation.shape == (2, 3, 3)
    assert grasp_state.contact_indices.shape == (2, 4)
    assert torch.isfinite(grasp_state.wrist_translation).all()


def test_mesh_object_signed_distance_and_normals_shapes():
    if not MESH_PATH.exists():
        return
    mesh_object = MeshObject(MESH_PATH)
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]], dtype=torch.float32)
    sdf = mesh_object.signed_distance(points)
    normals = mesh_object.normals(points)
    assert sdf.shape == (1, 2)
    assert normals.shape == (1, 2, 3)
    assert torch.isfinite(sdf).all()
    assert torch.isfinite(normals).all()


def test_mesh_object_uses_torchsdf_when_available():
    if not MESH_PATH.exists() or not HAS_TORCHSDF:
        return
    mesh_object = MeshObject(MESH_PATH)
    assert mesh_object._hull_face_verts is not None
    points = torch.tensor([[[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]], dtype=torch.float32)
    sdf = mesh_object.signed_distance(points)
    normals = mesh_object.normals(points)
    assert sdf.shape == (1, 2)
    assert normals.shape == (1, 2, 3)
    assert torch.isfinite(sdf).all()
    assert torch.isfinite(normals).all()


@pytest.mark.parametrize(
    "mesh_name",
    [
        "test_object.stl",
        "test_tube.stl",
    ],
)
def test_local_stl_assets_initialize_fingertip_grasps(mesh_name: str):
    mesh_path = Path("/home/haegu/minimal_graspqp/assets/objects") / mesh_name
    if not mesh_path.exists():
        return
    hand_model = ShadowHandModel.create(device="cpu", fingertips_only=True)
    mesh_object = MeshObject(mesh_path, scale=0.001)
    grasp_state = initialize_grasps_for_primitive(hand_model, mesh_object, batch_size=1, num_contacts=4)
    assert grasp_state.joint_values.shape == (1, 24)
    assert grasp_state.wrist_translation.shape == (1, 3)
    assert grasp_state.wrist_rotation.shape == (1, 3, 3)
    assert grasp_state.contact_indices.shape == (1, 4)
    assert torch.isfinite(grasp_state.wrist_translation).all()
