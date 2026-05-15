from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh as tm
from scipy.spatial.transform import Rotation

from minimal_graspqp.assets import resolve_shadow_hand_asset_dir
from minimal_graspqp.hands.shadow_hand import DEFAULT_JOINT_ORDER, DEFAULT_JOINT_STATE
from mujoco_grasp_smoke import _as_float_list, _load_object_visual_collision_meshes, _select_sample, _write_mesh
from minimal_graspqp.hands import MJCFShadowHandModel, ShadowHandModel


FINGER_CLOSE_DIRECTIONS = {
    "robot0_FFJ2": 1.0,
    "robot0_FFJ1": 1.0,
    "robot0_FFJ0": 1.0,
    "robot0_LFJ4": 1.0,
    "robot0_LFJ2": 1.0,
    "robot0_LFJ1": 1.0,
    "robot0_LFJ0": 1.0,
    "robot0_MFJ2": 1.0,
    "robot0_MFJ1": 1.0,
    "robot0_MFJ0": 1.0,
    "robot0_RFJ2": 1.0,
    "robot0_RFJ1": 1.0,
    "robot0_RFJ0": 1.0,
    "robot0_THJ3": 1.0,
    "robot0_THJ1": -1.0,
    "robot0_THJ0": -1.0,
}

JOINT_FINGERTIP_LINKS = {
    "robot0_FFJ2": "robot0_ffdistal",
    "robot0_FFJ1": "robot0_ffdistal",
    "robot0_FFJ0": "robot0_ffdistal",
    "robot0_MFJ2": "robot0_mfdistal",
    "robot0_MFJ1": "robot0_mfdistal",
    "robot0_MFJ0": "robot0_mfdistal",
    "robot0_RFJ2": "robot0_rfdistal",
    "robot0_RFJ1": "robot0_rfdistal",
    "robot0_RFJ0": "robot0_rfdistal",
    "robot0_LFJ4": "robot0_lfdistal",
    "robot0_LFJ2": "robot0_lfdistal",
    "robot0_LFJ1": "robot0_lfdistal",
    "robot0_LFJ0": "robot0_lfdistal",
    "robot0_THJ3": "robot0_thdistal",
    "robot0_THJ1": "robot0_thdistal",
    "robot0_THJ0": "robot0_thdistal",
}

FINGERTIP_LINK_NAMES = (
    "robot0_ffdistal",
    "robot0_mfdistal",
    "robot0_rfdistal",
    "robot0_lfdistal",
    "robot0_thdistal",
)

MUJOCO_DISTAL_BODY_NAMES = {
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
    "rh_thdistal",
}

MUJOCO_DISTAL_BODY_TO_SHORT = {
    "rh_ffdistal": "ff",
    "rh_mfdistal": "mf",
    "rh_rfdistal": "rf",
    "rh_lfdistal": "lf",
    "rh_thdistal": "th",
}

GRASPQP_TO_MUJOCO_BODY = {
    "robot0_palm": "rh_palm",
    "robot0_ffproximal": "rh_ffproximal",
    "robot0_ffmiddle": "rh_ffmiddle",
    "robot0_ffdistal": "rh_ffdistal",
    "robot0_mfproximal": "rh_mfproximal",
    "robot0_mfmiddle": "rh_mfmiddle",
    "robot0_mfdistal": "rh_mfdistal",
    "robot0_rfproximal": "rh_rfproximal",
    "robot0_rfmiddle": "rh_rfmiddle",
    "robot0_rfdistal": "rh_rfdistal",
    "robot0_lfproximal": "rh_lfproximal",
    "robot0_lfmiddle": "rh_lfmiddle",
    "robot0_lfdistal": "rh_lfdistal",
    "robot0_thproximal": "rh_thproximal",
    "robot0_thmiddle": "rh_thmiddle",
    "robot0_thdistal": "rh_thdistal",
}

FINGERTIP_PREFIXES = {
    "robot0_ffdistal": ("robot0_ffdistal_",),
    "robot0_mfdistal": ("robot0_mfdistal_",),
    "robot0_rfdistal": ("robot0_rfdistal_",),
    "robot0_lfdistal": ("robot0_lfdistal_",),
    "robot0_thdistal": ("robot0_thdistal_",),
}

LOCAL_MUJOCO_HAND_XML = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "mujoco_shadow_hand_in_hand_rolling"
    / "right_hand.xml"
)

ROBOT0_TO_RH_JOINT = {
    "robot0_WRJ1": "rh_WRJ2",
    "robot0_WRJ0": "rh_WRJ1",
    "robot0_FFJ3": "rh_FFJ4",
    "robot0_FFJ2": "rh_FFJ3",
    "robot0_FFJ1": "rh_FFJ2",
    "robot0_FFJ0": "rh_FFJ1",
    "robot0_LFJ4": "rh_LFJ5",
    "robot0_LFJ3": "rh_LFJ4",
    "robot0_LFJ2": "rh_LFJ3",
    "robot0_LFJ1": "rh_LFJ2",
    "robot0_LFJ0": "rh_LFJ1",
    "robot0_MFJ3": "rh_MFJ4",
    "robot0_MFJ2": "rh_MFJ3",
    "robot0_MFJ1": "rh_MFJ2",
    "robot0_MFJ0": "rh_MFJ1",
    "robot0_RFJ3": "rh_RFJ4",
    "robot0_RFJ2": "rh_RFJ3",
    "robot0_RFJ1": "rh_RFJ2",
    "robot0_RFJ0": "rh_RFJ1",
    "robot0_THJ4": "rh_THJ5",
    "robot0_THJ3": "rh_THJ4",
    "robot0_THJ2": "rh_THJ3",
    "robot0_THJ1": "rh_THJ2",
    "robot0_THJ0": "rh_THJ1",
}

ROBOT0_TO_RH_JOINT_SIGN = {
    "robot0_THJ1": -1.0,
    "robot0_THJ0": -1.0,
}


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _parse_vector(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(raw, sep=" ", dtype=np.float64)


def _quat_wxyz_from_rpy(rpy: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _transform_from_pos_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _parse_urdf(urdf_path: Path) -> tuple[ET.Element, dict[str, list[UrdfJoint]], dict[str, float], set[str]]:
    root = ET.fromstring(urdf_path.read_text())
    children_by_parent: dict[str, list[UrdfJoint]] = {}
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        limit = joint.find("limit")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parsed = UrdfJoint(
            name=joint.attrib["name"],
            joint_type=joint.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            xyz=_parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
            rpy=_parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
            axis=_parse_vector(axis.attrib.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0)),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        )
        children_by_parent.setdefault(parsed.parent, []).append(parsed)
        child_links.add(parsed.child)

    masses: dict[str, float] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        mass_node = inertial.find("mass") if inertial is not None else None
        if mass_node is not None:
            masses[link.attrib["name"]] = max(float(mass_node.attrib.get("value", 0.001)), 0.001)
    return root, children_by_parent, masses, child_links


def _mesh_asset_name(mesh_path: Path, scale: str, convex: bool, mesh_assets: dict[tuple[Path, str, bool], str]) -> str:
    key = (mesh_path.resolve(), scale, convex)
    if key not in mesh_assets:
        mesh_assets[key] = f"mesh_{len(mesh_assets)}"
    return mesh_assets[key]


def _scaled_mesh(mesh_path: Path, scale: str) -> tm.Trimesh:
    mesh = tm.load(mesh_path, force="mesh", process=False)
    scale_vec = np.fromstring(scale, sep=" ", dtype=np.float64)
    if scale_vec.size != 3:
        scale_vec = np.ones(3, dtype=np.float64)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale_vec.reshape(1, 3)
    return mesh


def _load_contact_meshes_by_link(asset_dir: Path) -> dict[str, Path]:
    contact_points_data = json.loads((asset_dir / "contact_points.json").read_text())
    mesh_root = asset_dir / "meshes"
    meshes_by_link: dict[str, Path] = {}
    for link_name, entries in contact_points_data.items():
        for entry in entries:
            if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str):
                meshes_by_link[link_name] = (mesh_root / entry[0]).resolve()
                break
    return meshes_by_link


def _resolve_mjcf_mesh_paths(root: ET.Element, xml_path: Path) -> None:
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir") if compiler is not None else None
    base_dir = xml_path.parent / meshdir if meshdir else xml_path.parent
    for mesh in root.findall("./asset/mesh"):
        filename = mesh.attrib.get("file")
        if filename is None:
            continue
        path = Path(filename)
        if not path.is_absolute():
            mesh.attrib["file"] = str((base_dir / path).resolve())
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]


def _name_unnamed_geoms(root: ET.Element) -> None:
    used = {geom.attrib["name"] for geom in root.iter("geom") if "name" in geom.attrib}

    def visit_body(body: ET.Element, prefix: str) -> None:
        body_name = body.attrib.get("name", prefix or "body")
        safe_body_name = body_name.replace("/", "_")
        geom_id = 0
        for child in body:
            if child.tag == "geom" and "name" not in child.attrib:
                while True:
                    name = f"{safe_body_name}_geom_{geom_id:02d}"
                    geom_id += 1
                    if name not in used:
                        break
                child.attrib["name"] = name
                used.add(name)
            elif child.tag == "body":
                visit_body(child, safe_body_name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    for body in worldbody.findall("body"):
        visit_body(body, "")


def _make_collision_geoms_visible(root: ET.Element, rgba: str) -> None:
    for geom in root.iter("geom"):
        geom_class = geom.attrib.get("class", "")
        if "collision" not in geom_class:
            continue
        geom.attrib["rgba"] = rgba
        geom.attrib["group"] = "1"


def _ensure_white_skybox(asset: ET.Element) -> None:
    if any(texture.attrib.get("type") == "skybox" for texture in asset.findall("texture")):
        return
    ET.SubElement(
        asset,
        "texture",
        name="white_skybox",
        type="skybox",
        builtin="gradient",
        rgb1="1 1 1",
        rgb2="1 1 1",
        width="512",
        height="512",
    )


def _rigid_alignment(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _graspqp_world_body_points(payload: dict, sample: int, joint_values: np.ndarray) -> dict[str, np.ndarray]:
    hand_info = payload.get("hand", {})
    if hand_info.get("model_type") == "mjcf":
        hand_kwargs = {
            "device": "cpu",
            "fingertips_only": bool(hand_info.get("fingertips_only", False)),
            "allowed_contact_links": hand_info.get("allowed_contact_links"),
        }
        if hand_info.get("mjcf_path"):
            hand_kwargs["mjcf_path"] = hand_info["mjcf_path"]
        hand_model = MJCFShadowHandModel.create(**hand_kwargs)
    else:
        hand_model = ShadowHandModel.create(
            asset_dir=hand_info.get("asset_dir"),
            device="cpu",
            fingertips_only=bool(hand_info.get("fingertips_only", False)),
            allowed_contact_links=hand_info.get("allowed_contact_links"),
        )
    joint_tensor = torch.as_tensor(joint_values, dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)
    fk = hand_model.forward_kinematics(joint_tensor)
    wrist_translation = payload["final_state"]["wrist_translation"][sample].detach().cpu().numpy()
    wrist_rotation = payload["final_state"]["wrist_rotation"][sample].detach().cpu().numpy()
    points = {}
    for link_name in GRASPQP_TO_MUJOCO_BODY:
        if link_name not in fk:
            continue
        local_position = fk[link_name][0, :3, 3].detach().cpu().numpy()
        points[link_name] = wrist_rotation @ local_position + wrist_translation
    return points


def _external_root_body_points(
    hand_xml: Path,
    joint_values: np.ndarray,
    root_body_name: str,
) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(hand_xml))
    data = mujoco.MjData(model)
    _set_external_hand_state(model, data, joint_values, np.zeros(3, dtype=np.float64))
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_id < 0:
        raise ValueError(f"Expected {root_body_name!r} body in {hand_xml}.")
    world_root = _transform_from_pos_matrix(data.xpos[root_id], data.xmat[root_id])
    root_inv = np.linalg.inv(world_root)
    points = {}
    for body_name in GRASPQP_TO_MUJOCO_BODY.values():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        world_point = np.ones(4, dtype=np.float64)
        world_point[:3] = data.xpos[body_id]
        points[body_name] = (root_inv @ world_point)[:3]
    return points


def _external_root_alignment_transform(
    payload: dict,
    sample: int,
    hand_xml: Path,
    joint_values: np.ndarray,
    root_body_name: str,
) -> np.ndarray:
    target_points_by_link = _graspqp_world_body_points(payload, sample, joint_values)
    source_points_by_body = _external_root_body_points(hand_xml, joint_values, root_body_name)
    source_points = []
    target_points = []
    for graspqp_link, mujoco_body in GRASPQP_TO_MUJOCO_BODY.items():
        if graspqp_link not in target_points_by_link or mujoco_body not in source_points_by_body:
            continue
        source_points.append(source_points_by_body[mujoco_body])
        target_points.append(target_points_by_link[graspqp_link])
    if len(source_points) < 3:
        raise ValueError("Need at least three corresponding hand bodies to align vendored MuJoCo hand.")
    return _rigid_alignment(np.asarray(source_points, dtype=np.float64), np.asarray(target_points, dtype=np.float64))


def _build_external_mujoco_hand_scene(
    payload: dict,
    sample: int,
    out_dir: Path,
    *,
    hand_xml: str | Path,
    mesh_path: str | None,
    hand_joint_values: np.ndarray,
    object_position: tuple[float, float, float] | None,
    timestep: float,
    gravity: tuple[float, float, float],
    friction: tuple[float, float, float],
    condim: int,
    object_mass: float,
    object_convex_hull: bool,
    floor_z: float | None,
    floor_size: float,
    actuator_kp: float | None = None,
    actuator_force: float | None = None,
    thumb_actuator_kp: float | None = None,
    thumb_actuator_force: float | None = None,
    joint_damping: float | None = None,
    contact_margin: float = 0.0,
    show_collision_geoms: bool = False,
) -> tuple[Path, np.ndarray, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    source_xml = Path(hand_xml).expanduser().resolve()
    root = ET.parse(source_xml).getroot()
    _resolve_mjcf_mesh_paths(root, source_xml)
    _name_unnamed_geoms(root)
    if show_collision_geoms:
        _make_collision_geoms_visible(root, "0.1 0.65 1 0.35")
    if (
        actuator_kp is not None
        or actuator_force is not None
        or thumb_actuator_kp is not None
        or thumb_actuator_force is not None
    ):
        actuator = root.find("actuator")
        if actuator is not None:
            for position in actuator.findall("position"):
                name = position.attrib.get("name", "")
                is_thumb = name.startswith("rh_A_THJ")
                kp = thumb_actuator_kp if is_thumb and thumb_actuator_kp is not None else actuator_kp
                force = thumb_actuator_force if is_thumb and thumb_actuator_force is not None else actuator_force
                if kp is not None and kp > 0.0:
                    position.attrib["kp"] = f"{kp:.9g}"
                if force is not None and force > 0.0:
                    position.attrib["forcerange"] = f"{-force:.9g} {force:.9g}"
    if joint_damping is not None and joint_damping >= 0.0:
        for joint in root.iter("joint"):
            joint.attrib["damping"] = f"{joint_damping:.9g}"

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.attrib["timestep"] = f"{timestep:.9g}"
    option.attrib["gravity"] = _as_float_list(gravity)

    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", solref="0.01 1", solimp="0.9 0.95 0.001", margin=f"{contact_margin:.9g}")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    _ensure_white_skybox(asset)
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    root_body = worldbody.find("body")
    if root_body is None or "name" not in root_body.attrib:
        raise ValueError(f"Expected a root hand body in {source_xml}.")

    desired_root = _external_root_alignment_transform(
        payload,
        sample,
        source_xml,
        hand_joint_values,
        root_body.attrib["name"],
    )
    root_body.attrib["pos"] = _as_float_list(desired_root[:3, 3])
    root_body.attrib["quat"] = _as_float_list(_quat_wxyz_from_matrix(desired_root[:3, :3]))

    if floor_z is not None and not any(node.attrib.get("name") == "floor" for node in worldbody.findall("geom")):
        ET.SubElement(
            worldbody,
            "geom",
            name="floor",
            type="plane",
            pos=f"0 0 {floor_z:.9g}",
            size=f"{floor_size:.9g} {floor_size:.9g} 0.005",
            rgba="0.45 0.45 0.45 1",
            friction=_as_float_list(friction),
            condim=str(condim),
        )

    object_visual_mesh, object_collision_mesh, object_center = _load_object_visual_collision_meshes(
        payload["primitive"],
        mesh_path,
        convex_hull=object_convex_hull,
    )
    object_visual_mesh_path = mesh_dir / "object_visual.obj"
    object_collision_mesh_path = mesh_dir / "object_collision.obj"
    _write_mesh(object_visual_mesh_path, np.asarray(object_visual_mesh.vertices), np.asarray(object_visual_mesh.faces))
    _write_mesh(object_collision_mesh_path, np.asarray(object_collision_mesh.vertices), np.asarray(object_collision_mesh.faces))
    ET.SubElement(asset, "mesh", name="object_visual_mesh", file=str(object_visual_mesh_path.resolve()))
    ET.SubElement(asset, "mesh", name="object_collision_mesh", file=str(object_collision_mesh_path.resolve()))
    if object_position is not None:
        object_center = np.asarray(object_position, dtype=np.float64)
    object_body = ET.SubElement(worldbody, "body", name="object", pos=_as_float_list(object_center))
    ET.SubElement(object_body, "freejoint", name="object_free")
    ET.SubElement(
        object_body,
        "geom",
        name="object_visual_geom",
        type="mesh",
        mesh="object_visual_mesh",
        contype="0",
        conaffinity="0",
        rgba="0.05 0.7 0.12 1",
    )
    ET.SubElement(
        object_body,
        "geom",
        name="object_geom",
        type="mesh",
        mesh="object_collision_mesh",
        mass=f"{object_mass:.9g}",
        friction=_as_float_list(friction),
        condim=str(condim),
        rgba="0.05 0.7 0.12 0.05",
    )

    xml_path = out_dir / "external_hand_scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_names = [name for name in ROBOT0_TO_RH_JOINT.values() if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0]
    return xml_path, object_center, joint_names


def _add_link_geoms(
    asset: ET.Element,
    body: ET.Element,
    link_node: ET.Element,
    asset_dir: Path,
    generated_mesh_dir: Path,
    mesh_assets: dict[tuple[Path, str, bool], str],
    friction: tuple[float, float, float],
    condim: int,
    hand_convex_collision: bool,
    contact_meshes_by_link: dict[str, Path],
    contact_spheres_by_link: dict[str, list[np.ndarray]],
    contact_sphere_radius: float,
) -> None:
    def add_element(element: ET.Element, element_id: int, *, visual: bool) -> None:
        geometry = element.find("geometry")
        if geometry is None:
            return
        origin = element.find("origin")
        xyz = _parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
        suffix = "visual" if visual else "collision"
        common = {
            "name": f"{link_node.attrib['name']}_{suffix}_{element_id}",
            "pos": _as_float_list(xyz),
            "quat": _as_float_list(_quat_wxyz_from_rpy(rpy)),
            "rgba": "0.72 0.86 0.95 0.7",
            "group": "1",
        }
        if visual:
            common["contype"] = "0"
            common["conaffinity"] = "0"
        else:
            common["friction"] = _as_float_list(friction)
            common["condim"] = str(condim)

        mesh = geometry.find("mesh")
        if mesh is not None:
            filename = mesh.attrib["filename"].replace("package://", "")
            mesh_path = (asset_dir / filename).resolve()
            scale = mesh.attrib.get("scale", "1 1 1")
            use_convex = hand_convex_collision and not visual
            mesh_name = _mesh_asset_name(mesh_path, scale, use_convex, mesh_assets)
            if not any(node.attrib.get("name") == mesh_name for node in asset.findall("mesh")):
                if use_convex:
                    hull = _scaled_mesh(mesh_path, scale).convex_hull
                    generated_path = generated_mesh_dir / f"{mesh_name}.obj"
                    _write_mesh(generated_path, np.asarray(hull.vertices), np.asarray(hull.faces))
                    ET.SubElement(asset, "mesh", name=mesh_name, file=str(generated_path.resolve()))
                else:
                    ET.SubElement(asset, "mesh", name=mesh_name, file=str(mesh_path), scale=scale)
            ET.SubElement(body, "geom", type="mesh", mesh=mesh_name, **common)
            return

        box = geometry.find("box")
        if box is not None:
            size = 0.5 * np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
            ET.SubElement(body, "geom", type="box", size=_as_float_list(size), **common)
            return

        sphere = geometry.find("sphere")
        if sphere is not None:
            ET.SubElement(body, "geom", type="sphere", size=sphere.attrib["radius"], **common)
            return

        cylinder = geometry.find("cylinder")
        if cylinder is not None:
            size = f"{float(cylinder.attrib['radius']):.9g} {0.5 * float(cylinder.attrib['length']):.9g}"
            ET.SubElement(body, "geom", type="cylinder", size=size, **common)

    link_name = link_node.attrib["name"]
    contact_mesh_path = contact_meshes_by_link.get(link_name)
    if contact_mesh_path is not None:
        mesh_name = _mesh_asset_name(contact_mesh_path, "1 1 1", False, mesh_assets)
        if not any(node.attrib.get("name") == mesh_name for node in asset.findall("mesh")):
            ET.SubElement(asset, "mesh", name=mesh_name, file=str(contact_mesh_path))
        ET.SubElement(
            body,
            "geom",
            name=f"{link_name}_contact_mesh",
            type="mesh",
            mesh=mesh_name,
            friction=_as_float_list(friction),
            condim=str(condim),
            rgba="0.72 0.86 0.95 0.7",
            group="1",
        )
    else:
        collision_elements = link_node.findall("collision")
        if not collision_elements:
            collision_elements = link_node.findall("visual")
        for element_id, element in enumerate(collision_elements):
            add_element(element, element_id, visual=False)

    if contact_sphere_radius > 0.0:
        for sphere_id, point in enumerate(contact_spheres_by_link.get(link_name, [])):
            ET.SubElement(
                body,
                "geom",
                name=f"{link_name}_contact_sphere_{sphere_id}",
                type="sphere",
                pos=_as_float_list(point),
                size=f"{contact_sphere_radius:.9g}",
                friction=_as_float_list(friction),
                condim=str(condim),
                rgba="1 0.08 0.08 0.55",
            )


def _add_link_body(
    parent_body: ET.Element,
    asset: ET.Element,
    link_nodes: dict[str, ET.Element],
    children_by_parent: dict[str, list[UrdfJoint]],
    masses: dict[str, float],
    asset_dir: Path,
    generated_mesh_dir: Path,
    mesh_assets: dict[tuple[Path, str, bool], str],
    joint: UrdfJoint,
    friction: tuple[float, float, float],
    condim: int,
    joint_damping: float,
    hand_convex_collision: bool,
    contact_meshes_by_link: dict[str, Path],
    contact_spheres_by_link: dict[str, list[np.ndarray]],
    contact_sphere_radius: float,
) -> None:
    body = ET.SubElement(
        parent_body,
        "body",
        name=joint.child,
        pos=_as_float_list(joint.xyz),
        quat=_as_float_list(_quat_wxyz_from_rpy(joint.rpy)),
    )
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=f"{masses.get(joint.child, 0.001):.9g}",
        diaginertia="1e-6 1e-6 1e-6",
    )
    if joint.joint_type in {"revolute", "continuous"}:
        attrs = {
            "name": joint.name,
            "type": "hinge",
            "axis": _as_float_list(joint.axis),
            "damping": f"{joint_damping:.9g}",
            "armature": "1e-5",
            "limited": "true",
        }
        if joint.lower is not None and joint.upper is not None:
            attrs["range"] = f"{joint.lower:.9g} {joint.upper:.9g}"
        ET.SubElement(body, "joint", **attrs)

    link_node = link_nodes.get(joint.child)
    if link_node is not None:
        _add_link_geoms(
            asset,
            body,
            link_node,
            asset_dir,
            generated_mesh_dir,
            mesh_assets,
            friction,
            condim,
            hand_convex_collision,
            contact_meshes_by_link,
            contact_spheres_by_link,
            contact_sphere_radius,
        )

    for child_joint in children_by_parent.get(joint.child, []):
        _add_link_body(
            body,
            asset,
            link_nodes,
            children_by_parent,
            masses,
            asset_dir,
            generated_mesh_dir,
            mesh_assets,
            child_joint,
            friction,
            condim,
            joint_damping,
            hand_convex_collision,
            contact_meshes_by_link,
            contact_spheres_by_link,
            contact_sphere_radius,
        )


def _joint_limits(children_by_parent: dict[str, list[UrdfJoint]]) -> dict[str, tuple[float, float]]:
    limits = {}
    for joints in children_by_parent.values():
        for joint in joints:
            if joint.joint_type in {"revolute", "continuous"} and joint.lower is not None and joint.upper is not None:
                limits[joint.name] = (joint.lower, joint.upper)
    return limits


def _apply_joint_offset(
    joint_values: np.ndarray,
    joint_name: str,
    offset: float,
    limits: dict[str, tuple[float, float]],
) -> None:
    if abs(offset) <= 0.0:
        return
    try:
        joint_id = DEFAULT_JOINT_ORDER.index(joint_name)
    except ValueError as exc:
        raise ValueError(f"Unknown joint for offset: {joint_name}") from exc
    lower, upper = limits[joint_name]
    before = float(joint_values[joint_id])
    joint_values[joint_id] = np.clip(before + offset, lower, upper)
    print(
        f"Applied joint offset: {joint_name} {before:.6g} -> {joint_values[joint_id]:.6g} "
        f"(offset={joint_values[joint_id] - before:.6g})"
    )


def _preload_targets(joint_values: np.ndarray, limits: dict[str, tuple[float, float]], preload_fraction: float) -> np.ndarray:
    targets = joint_values.copy()
    if preload_fraction <= 0.0:
        return targets
    for joint_id, joint_name in enumerate(DEFAULT_JOINT_ORDER):
        direction = FINGER_CLOSE_DIRECTIONS.get(joint_name)
        bounds = limits.get(joint_name)
        if direction is None or bounds is None:
            continue
        lower, upper = bounds
        if direction > 0:
            targets[joint_id] = targets[joint_id] + preload_fraction * max(upper - targets[joint_id], 0.0)
        else:
            targets[joint_id] = targets[joint_id] - preload_fraction * max(targets[joint_id] - lower, 0.0)
        targets[joint_id] = np.clip(targets[joint_id], lower, upper)
    return targets


def _open_hand_targets(joint_values: np.ndarray, limits: dict[str, tuple[float, float]], open_fraction: float) -> np.ndarray:
    targets = joint_values.copy()
    if open_fraction <= 0.0:
        return targets
    for joint_id, joint_name in enumerate(DEFAULT_JOINT_ORDER):
        direction = FINGER_CLOSE_DIRECTIONS.get(joint_name)
        bounds = limits.get(joint_name)
        if direction is None or bounds is None:
            continue
        lower, upper = bounds
        if direction > 0:
            targets[joint_id] = targets[joint_id] - open_fraction * max(targets[joint_id] - lower, 0.0)
        else:
            targets[joint_id] = targets[joint_id] + open_fraction * max(upper - targets[joint_id], 0.0)
        targets[joint_id] = np.clip(targets[joint_id], lower, upper)
    return targets


def _selected_contact_spheres(payload: dict, sample: int, radius: float) -> dict[str, list[np.ndarray]]:
    if radius <= 0.0:
        return {}
    hand_info = payload.get("hand", {})
    if hand_info.get("model_type") == "mjcf":
        hand_kwargs = {
            "device": "cpu",
            "fingertips_only": bool(hand_info.get("fingertips_only", False)),
            "allowed_contact_links": hand_info.get("allowed_contact_links"),
        }
        if hand_info.get("mjcf_path"):
            hand_kwargs["mjcf_path"] = hand_info["mjcf_path"]
        model = MJCFShadowHandModel.create(**hand_kwargs)
    else:
        model = ShadowHandModel.create(
            device="cpu",
            fingertips_only=bool(hand_info.get("fingertips_only", False)),
            allowed_contact_links=hand_info.get("allowed_contact_links"),
        )
    indices = payload["final_state"]["contact_indices"][sample].detach().cpu().reshape(-1).tolist()
    grouped: dict[str, list[np.ndarray]] = {}
    for index in indices:
        link_name = model.metadata.contact_candidate_links[int(index)]
        point = model.metadata.contact_candidate_points[int(index)].detach().cpu().numpy().astype(np.float64)
        grouped.setdefault(link_name, []).append(point)
    return grouped


def _build_dynamic_scene(
    payload: dict,
    sample: int,
    out_dir: Path,
    *,
    mesh_path: str | None,
    timestep: float,
    gravity: tuple[float, float, float],
    friction: tuple[float, float, float],
    condim: int,
    object_mass: float,
    object_convex_hull: bool,
    fix_object: bool,
    floor_z: float | None,
    floor_size: float,
    actuator_kp: float,
    actuator_force: float,
    thumb_actuator_kp: float | None,
    thumb_actuator_force: float | None,
    joint_damping: float,
    hand_convex_collision: bool,
    full_contact_mesh: bool,
    contact_sphere_radius: float,
) -> tuple[Path, np.ndarray, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    asset_dir = resolve_shadow_hand_asset_dir()
    urdf_path = asset_dir / "shadow_hand.urdf"
    urdf_root, children_by_parent, masses, child_links = _parse_urdf(urdf_path)
    link_nodes = {link.attrib["name"]: link for link in urdf_root.findall("link")}
    root_links = [link.attrib["name"] for link in urdf_root.findall("link") if link.attrib["name"] not in child_links]
    if root_links != ["root"]:
        raise ValueError(f"Unexpected URDF root links: {root_links}")

    final_state = payload["final_state"]
    wrist_translation = final_state["wrist_translation"][sample].detach().cpu().numpy()
    wrist_rotation = final_state["wrist_rotation"][sample].detach().cpu().numpy()
    contact_spheres_by_link = _selected_contact_spheres(payload, sample, contact_sphere_radius)
    contact_meshes_by_link = _load_contact_meshes_by_link(asset_dir) if full_contact_mesh else {}
    limits = _joint_limits(children_by_parent)

    root = ET.Element("mujoco", model="minimal_graspqp_dynamic_shadow_hand")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", balanceinertia="true")
    ET.SubElement(root, "option", timestep=f"{timestep:.9g}", gravity=_as_float_list(gravity), integrator="implicitfast")
    ET.SubElement(root, "size", njmax="3000", nconmax="800")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", solref="0.01 1", solimp="0.9 0.95 0.001", margin="0.0005")
    asset = ET.SubElement(root, "asset")
    _ensure_white_skybox(asset)
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", name="key", pos="0 -0.4 1.0", dir="0 0 -1")
    if floor_z is not None:
        ET.SubElement(
            worldbody,
            "geom",
            name="floor",
            type="plane",
            pos=f"0 0 {floor_z:.9g}",
            size=f"{floor_size:.9g} {floor_size:.9g} 0.005",
            rgba="0.45 0.45 0.45 1",
            friction=_as_float_list(friction),
            condim=str(condim),
        )

    hand_root = ET.SubElement(
        worldbody,
        "body",
        name="hand_root",
        pos=_as_float_list(wrist_translation),
        quat=_as_float_list(_quat_wxyz_from_matrix(wrist_rotation)),
    )
    ET.SubElement(hand_root, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
    mesh_assets: dict[tuple[Path, str, bool], str] = {}
    for child_joint in children_by_parent["root"]:
        _add_link_body(
            hand_root,
            asset,
            link_nodes,
            children_by_parent,
            masses,
            asset_dir,
            mesh_dir,
            mesh_assets,
            child_joint,
            friction,
            condim,
            joint_damping,
            hand_convex_collision,
            contact_meshes_by_link,
            contact_spheres_by_link,
            contact_sphere_radius,
        )

    object_visual_mesh, object_collision_mesh, object_center = _load_object_visual_collision_meshes(
        payload["primitive"],
        mesh_path,
        convex_hull=object_convex_hull,
    )
    object_visual_mesh_path = mesh_dir / "object_visual.obj"
    object_collision_mesh_path = mesh_dir / "object_collision.obj"
    _write_mesh(object_visual_mesh_path, np.asarray(object_visual_mesh.vertices), np.asarray(object_visual_mesh.faces))
    _write_mesh(object_collision_mesh_path, np.asarray(object_collision_mesh.vertices), np.asarray(object_collision_mesh.faces))
    ET.SubElement(asset, "mesh", name="object_visual_mesh", file=str(object_visual_mesh_path.resolve()))
    ET.SubElement(asset, "mesh", name="object_collision_mesh", file=str(object_collision_mesh_path.resolve()))
    object_body = ET.SubElement(worldbody, "body", name="object", pos=_as_float_list(object_center))
    if not fix_object:
        ET.SubElement(object_body, "freejoint", name="object_free")
    ET.SubElement(
        object_body,
        "geom",
        name="object_visual_geom",
        type="mesh",
        mesh="object_collision_mesh",
        contype="0",
        conaffinity="0",
        rgba="0.05 0.7 0.12 1",
    )
    ET.SubElement(
        object_body,
        "geom",
        name="object_geom",
        type="mesh",
        mesh="object_collision_mesh",
        mass=f"{object_mass:.9g}",
        friction=_as_float_list(friction),
        condim=str(condim),
        rgba="0.05 0.7 0.12 0.05",
    )

    actuator = ET.SubElement(root, "actuator")
    for joint_name in DEFAULT_JOINT_ORDER:
        is_thumb = joint_name.startswith("robot0_THJ")
        kp = thumb_actuator_kp if is_thumb and thumb_actuator_kp is not None else actuator_kp
        force = thumb_actuator_force if is_thumb and thumb_actuator_force is not None else actuator_force
        attrs = {
            "name": f"{joint_name}_pos",
            "joint": joint_name,
            "kp": f"{kp:.9g}",
            "forcerange": f"{-force:.9g} {force:.9g}",
        }
        if joint_name in limits:
            lower, upper = limits[joint_name]
            attrs["ctrllimited"] = "true"
            attrs["ctrlrange"] = f"{lower:.9g} {upper:.9g}"
        ET.SubElement(actuator, "position", **attrs)

    xml_path = out_dir / "dynamic_scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path, object_center, DEFAULT_JOINT_ORDER


def _set_dynamic_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    joint_values: np.ndarray,
    ctrl_values: np.ndarray,
    object_center: np.ndarray,
) -> None:
    for joint_name, joint_value in zip(joint_names, joint_values):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint missing from MuJoCo model: {joint_name}")
        data.qpos[model.jnt_qposadr[joint_id]] = joint_value

    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if object_joint >= 0:
        qpos_adr = model.jnt_qposadr[object_joint]
        data.qpos[qpos_adr : qpos_adr + 3] = object_center
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    data.ctrl[:] = ctrl_values
    mujoco.mj_forward(model, data)


def _set_external_hand_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot0_joint_values: np.ndarray,
    object_center: np.ndarray,
) -> None:
    _set_external_hand_joints(model, data, robot0_joint_values)

    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if object_joint >= 0:
        qpos_adr = model.jnt_qposadr[object_joint]
        data.qpos[qpos_adr : qpos_adr + 3] = object_center
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _set_external_hand_joints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot0_joint_values: np.ndarray,
) -> None:
    for robot0_name, rh_name in ROBOT0_TO_RH_JOINT.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rh_name)
        if joint_id < 0:
            continue
        value = _external_joint_value(robot0_name, robot0_joint_values)
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"rh_A_{rh_name.removeprefix('rh_')}")
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value


def _external_joint_value(robot0_name: str, robot0_joint_values: np.ndarray) -> float:
    value = float(robot0_joint_values[DEFAULT_JOINT_ORDER.index(robot0_name)])
    return value * ROBOT0_TO_RH_JOINT_SIGN.get(robot0_name, 1.0)


def _reset_object_pose(model: mujoco.MjModel, data: mujoco.MjData, object_center: np.ndarray) -> None:
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if object_joint < 0:
        return
    qpos_adr = model.jnt_qposadr[object_joint]
    qvel_adr = model.jnt_dofadr[object_joint]
    data.qpos[qpos_adr : qpos_adr + 3] = object_center
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _set_object_joint_locked(model: mujoco.MjModel, data: mujoco.MjData, locked: bool) -> None:
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    if object_joint < 0:
        return
    dof_adr = model.jnt_dofadr[object_joint]
    qvel_adr = model.jnt_dofadr[object_joint]
    if locked:
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        model.dof_damping[dof_adr : dof_adr + 6] = 1e6
        model.dof_frictionloss[dof_adr : dof_adr + 6] = 1e6
    else:
        model.dof_damping[dof_adr : dof_adr + 6] = 0.0
        model.dof_frictionloss[dof_adr : dof_adr + 6] = 0.0


def _object_contact_force_norm(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, int]:
    total = 0.0
    count = 0
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if name1 == "floor" or name2 == "floor":
            continue
        if name1 != "object_geom" and name2 != "object_geom":
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        total += float(np.linalg.norm(force[:3]))
        count += 1
    return total, count


def _object_distal_contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int]:
    distal_links = set()
    distal_contacts = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "",
        ]
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or "",
        ]
        if "object_geom" not in geom_names:
            continue
        for body_name in body_names:
            if body_name in MUJOCO_DISTAL_BODY_NAMES:
                distal_links.add(body_name)
                distal_contacts += 1
                break
    return len(distal_links), distal_contacts


def _object_distal_contact_forces_by_body(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    forces = {body_name: 0.0 for body_name in MUJOCO_DISTAL_BODY_NAMES}
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "",
        ]
        if "object_geom" not in geom_names:
            continue
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or "",
        ]
        for body_name in body_names:
            if body_name in forces:
                mujoco.mj_contactForce(model, data, contact_id, force)
                forces[body_name] += float(np.linalg.norm(force[:3]))
                break
    return forces


def _fingertip_object_contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    forces = {name: 0.0 for name in FINGERTIP_LINK_NAMES}
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if name1 == "floor" or name2 == "floor":
            continue
        if name1 != "object_geom" and name2 != "object_geom":
            continue
        other = name2 if name1 == "object_geom" else name1
        mujoco.mj_contactForce(model, data, contact_id, force)
        norm = float(np.linalg.norm(force[:3]))
        for link_name, prefixes in FINGERTIP_PREFIXES.items():
            if any(other.startswith(prefix) for prefix in prefixes):
                forces[link_name] += norm
                break
    return forces


def _joint_tracking_error(model: mujoco.MjModel, data: mujoco.MjData, joint_names: list[str], target: np.ndarray) -> float:
    values = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        values.append(float(data.qpos[model.jnt_qposadr[joint_id]]))
    return float(np.max(np.abs(np.asarray(values) - target)))


def _run_dynamic_smoke(
    xml_path: Path,
    object_center: np.ndarray,
    joint_names: list[str],
    joint_values: np.ndarray,
    ctrl_values: np.ndarray,
    steps: int,
    preload_steps: int,
    freeze_object_during_preload: bool,
    zero_gravity_during_preload: bool,
) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_dynamic_state(model, data, joint_names, joint_values, ctrl_values, object_center)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    initial_error = _joint_tracking_error(model, data, joint_names, ctrl_values)
    original_gravity = model.opt.gravity.copy()

    if preload_steps > 0:
        if zero_gravity_during_preload:
            model.opt.gravity[:] = 0.0
        for _ in range(preload_steps):
            data.ctrl[:] = ctrl_values
            if freeze_object_during_preload:
                _reset_object_pose(model, data, object_center)
            mujoco.mj_step(model, data)
        if freeze_object_during_preload:
            _reset_object_pose(model, data, object_center)
        model.opt.gravity[:] = original_gravity
        mujoco.mj_forward(model, data)

    release_error = _joint_tracking_error(model, data, joint_names, ctrl_values)
    initial_pos = data.xpos[object_body].copy()
    contact_counts = []
    max_error = max(initial_error, release_error)
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.ctrl[:] = ctrl_values
        max_error = max(max_error, _joint_tracking_error(model, data, joint_names, ctrl_values))
        object_contacts = 0
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            if name1 == "object_geom" or name2 == "object_geom":
                object_contacts += 1
        contact_counts.append(object_contacts)
    final_pos = data.xpos[object_body].copy()
    displacement = final_pos - initial_pos
    return {
        "nq": float(model.nq),
        "nv": float(model.nv),
        "nu": float(model.nu),
        "initial_z": float(initial_pos[2]),
        "final_z": float(final_pos[2]),
        "z_drop": float(initial_pos[2] - final_pos[2]),
        "displacement_norm": float(np.linalg.norm(displacement)),
        "initial_joint_error": initial_error,
        "release_joint_error": release_error,
        "max_joint_error": max_error,
        "target_delta_norm": float(np.linalg.norm(ctrl_values - joint_values)),
        "target_delta_max": float(np.max(np.abs(ctrl_values - joint_values))),
        "preload_steps": float(preload_steps),
        "mean_object_contacts": float(np.mean(contact_counts) if contact_counts else 0.0),
        "final_object_contacts": float(contact_counts[-1] if contact_counts else 0.0),
    }


def _run_external_dynamic_smoke(
    xml_path: Path,
    object_center: np.ndarray,
    joint_values: np.ndarray,
    steps: int,
    preload_steps: int,
    freeze_object_during_preload: bool,
    zero_gravity_during_preload: bool,
    hold_hand_qpos: bool = False,
) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_external_hand_state(model, data, joint_values, object_center)
    object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    original_gravity = model.opt.gravity.copy()

    if preload_steps > 0:
        if zero_gravity_during_preload:
            model.opt.gravity[:] = 0.0
        for _ in range(preload_steps):
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
            if freeze_object_during_preload:
                _reset_object_pose(model, data, object_center)
            mujoco.mj_step(model, data)
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
                mujoco.mj_forward(model, data)
        if freeze_object_during_preload:
            _reset_object_pose(model, data, object_center)
        model.opt.gravity[:] = original_gravity
        mujoco.mj_forward(model, data)

    initial_pos = data.xpos[object_body].copy()
    contact_counts = []
    contact_forces = []
    distal_link_counts = []
    distal_contact_counts = []
    distal_force_trace = {body_name: [] for body_name in MUJOCO_DISTAL_BODY_NAMES}
    first_zero_contact_step = -1
    for _ in range(steps):
        if hold_hand_qpos:
            _set_external_hand_joints(model, data, joint_values)
            mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        if hold_hand_qpos:
            _set_external_hand_joints(model, data, joint_values)
            mujoco.mj_forward(model, data)
        contact_force, contact_count = _object_contact_force_norm(model, data)
        distal_link_count, distal_contact_count = _object_distal_contact_summary(model, data)
        distal_forces = _object_distal_contact_forces_by_body(model, data)
        if contact_count == 0 and first_zero_contact_step < 0:
            first_zero_contact_step = len(contact_counts)
        contact_counts.append(contact_count)
        contact_forces.append(contact_force)
        distal_link_counts.append(distal_link_count)
        distal_contact_counts.append(distal_contact_count)
        for body_name, value in distal_forces.items():
            distal_force_trace[body_name].append(value)

    final_pos = data.xpos[object_body].copy()
    displacement = final_pos - initial_pos
    metrics = {
        "nq": float(model.nq),
        "nv": float(model.nv),
        "nu": float(model.nu),
        "initial_z": float(initial_pos[2]),
        "final_z": float(final_pos[2]),
        "z_drop": float(initial_pos[2] - final_pos[2]),
        "displacement_norm": float(np.linalg.norm(displacement)),
        "preload_steps": float(preload_steps),
        "hold_hand_qpos": float(hold_hand_qpos),
        "mean_object_contacts": float(np.mean(contact_counts) if contact_counts else 0.0),
        "final_object_contacts": float(contact_counts[-1] if contact_counts else 0.0),
        "min_object_contacts": float(np.min(contact_counts) if contact_counts else 0.0),
        "mean_contact_force": float(np.mean(contact_forces) if contact_forces else 0.0),
        "final_contact_force": float(contact_forces[-1] if contact_forces else 0.0),
        "min_contact_force": float(np.min(contact_forces) if contact_forces else 0.0),
        "mean_distal_contact_links": float(np.mean(distal_link_counts) if distal_link_counts else 0.0),
        "final_distal_contact_links": float(distal_link_counts[-1] if distal_link_counts else 0.0),
        "min_distal_contact_links": float(np.min(distal_link_counts) if distal_link_counts else 0.0),
        "mean_distal_contacts": float(np.mean(distal_contact_counts) if distal_contact_counts else 0.0),
        "final_distal_contacts": float(distal_contact_counts[-1] if distal_contact_counts else 0.0),
        "first_zero_contact_step": float(first_zero_contact_step),
    }
    for body_name, short_name in MUJOCO_DISTAL_BODY_TO_SHORT.items():
        values = distal_force_trace[body_name]
        metrics[f"mean_{short_name}_force"] = float(np.mean(values) if values else 0.0)
        metrics[f"final_{short_name}_force"] = float(values[-1] if values else 0.0)
    return metrics


def _launch_external_dynamic_viewer(
    xml_path: Path,
    object_center: np.ndarray,
    joint_values: np.ndarray,
    *,
    preload_steps: int,
    freeze_object_during_preload: bool,
    zero_gravity_during_preload: bool,
    max_viewer_steps: int,
    show_contact_points: bool,
    hold_hand_qpos: bool,
) -> None:
    import time
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_external_hand_state(model, data, joint_values, object_center)
    original_gravity = model.opt.gravity.copy()

    if preload_steps > 0:
        if zero_gravity_during_preload:
            model.opt.gravity[:] = 0.0
        for _ in range(preload_steps):
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
            if freeze_object_during_preload:
                _reset_object_pose(model, data, object_center)
            mujoco.mj_step(model, data)
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
                mujoco.mj_forward(model, data)
        if freeze_object_during_preload:
            _reset_object_pose(model, data, object_center)
        model.opt.gravity[:] = original_gravity
        mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if show_contact_points:
            for flag_name in ("mjVIS_CONTACTPOINT", "mjVIS_CONTACTFORCE"):
                flag = getattr(mujoco.mjtVisFlag, flag_name, None)
                if flag is not None:
                    viewer.opt.flags[flag] = 1
        step = 0
        while viewer.is_running() and (max_viewer_steps <= 0 or step < max_viewer_steps):
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
                mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            if hold_hand_qpos:
                _set_external_hand_joints(model, data, joint_values)
                mujoco.mj_forward(model, data)
            if show_contact_points:
                viewer.user_scn.ngeom = 0
                marker_mat = np.eye(3, dtype=np.float64).reshape(-1)
                marker_size = np.array([0.0025, 0.0, 0.0], dtype=np.float64)
                marker_rgba = np.array([1.0, 0.05, 0.0, 1.0], dtype=np.float32)
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                    geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                    if "object_geom" not in (geom1, geom2):
                        continue
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        marker_size,
                        contact.pos,
                        marker_mat,
                        marker_rgba,
                    )
                    viewer.user_scn.ngeom += 1
            if step % 60 == 0:
                contact_force, contact_count = _object_contact_force_norm(model, data)
                distal_links, distal_contacts = _object_distal_contact_summary(model, data)
                distal_forces = _object_distal_contact_forces_by_body(model, data)
                force_text = " ".join(
                    f"{MUJOCO_DISTAL_BODY_TO_SHORT[name]}={value:.2f}"
                    for name, value in sorted(distal_forces.items())
                )
                print(
                    f"viewer step={step} contact_force={contact_force:.6g} "
                    f"contacts={contact_count} distal_links={distal_links} "
                    f"distal_contacts={distal_contacts} {force_text}",
                    flush=True,
                )
            viewer.sync()
            time.sleep(model.opt.timestep)
            step += 1


def _launch_dynamic_viewer(
    xml_path: Path,
    object_center: np.ndarray,
    joint_names: list[str],
    joint_values: np.ndarray,
    ctrl_values: np.ndarray,
    *,
    preload_steps: int,
    freeze_object_during_preload: bool,
    zero_gravity_during_preload: bool,
) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_dynamic_state(model, data, joint_names, joint_values, ctrl_values, object_center)
    original_gravity = model.opt.gravity.copy()
    if preload_steps > 0:
        if zero_gravity_during_preload:
            model.opt.gravity[:] = 0.0
        for _ in range(preload_steps):
            data.ctrl[:] = ctrl_values
            if freeze_object_during_preload:
                _reset_object_pose(model, data, object_center)
            mujoco.mj_step(model, data)
        if freeze_object_during_preload:
            _reset_object_pose(model, data, object_center)
        model.opt.gravity[:] = original_gravity
        mujoco.mj_forward(model, data)
    data.ctrl[:] = ctrl_values
    mujoco.viewer.launch(model, data)


def _launch_ramp_release_viewer(
    xml_path: Path,
    object_center: np.ndarray,
    joint_names: list[str],
    open_values: np.ndarray,
    target_values: np.ndarray,
    *,
    ramp_steps: int,
    hold_steps: int,
    force_threshold: float,
    release_after_force: bool,
    per_finger_force_threshold: float,
    min_force_fingers: int,
    close_until_force: bool,
    stop_fingers_on_force: bool,
    close_rate: float,
    max_viewer_steps: int,
) -> None:
    import mujoco.viewer
    import time

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _set_dynamic_state(model, data, joint_names, open_values, open_values, object_center)
    _set_object_joint_locked(model, data, True)

    released = False
    force_hit_step = -1
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        current_ctrl = open_values.copy()
        while viewer.is_running() and (max_viewer_steps <= 0 or step < max_viewer_steps):
            fingertip_forces = _fingertip_object_contact_forces(model, data)
            active_fingers = sum(force >= per_finger_force_threshold for force in fingertip_forces.values())
            contact_force, contact_count = _object_contact_force_norm(model, data)
            force_ready = (
                active_fingers >= min_force_fingers
                and contact_count > 0
                and contact_force >= force_threshold
            )
            if close_until_force:
                for joint_id, joint_name in enumerate(DEFAULT_JOINT_ORDER):
                    direction = FINGER_CLOSE_DIRECTIONS.get(joint_name)
                    if direction is None:
                        current_ctrl[joint_id] = target_values[joint_id]
                        continue
                    fingertip_link = JOINT_FINGERTIP_LINKS.get(joint_name)
                    finger_ready = (
                        stop_fingers_on_force
                        and fingertip_link is not None
                        and fingertip_forces[fingertip_link] >= per_finger_force_threshold
                    )
                    if not force_ready and not finger_ready:
                        delta = close_rate * direction
                        if direction > 0:
                            current_ctrl[joint_id] = min(target_values[joint_id], current_ctrl[joint_id] + delta)
                        else:
                            current_ctrl[joint_id] = max(target_values[joint_id], current_ctrl[joint_id] + delta)
                    else:
                        current_ctrl[joint_id] = current_ctrl[joint_id]
                ctrl = current_ctrl
                alpha = float(np.mean(np.isclose(ctrl, target_values, atol=1e-4)))
            else:
                if ramp_steps > 0 and step < ramp_steps:
                    alpha = (step + 1) / ramp_steps
                else:
                    alpha = 1.0
                ctrl = (1.0 - alpha) * open_values + alpha * target_values
            data.ctrl[:] = ctrl

            if not released:
                _reset_object_pose(model, data, object_center)
            mujoco.mj_step(model, data)

            contact_force, contact_count = _object_contact_force_norm(model, data)
            fingertip_forces = _fingertip_object_contact_forces(model, data)
            active_fingers = sum(force >= per_finger_force_threshold for force in fingertip_forces.values())
            should_release = (
                release_after_force
                and not released
                and step >= ramp_steps + hold_steps
                and contact_count > 0
                and contact_force >= force_threshold
                and active_fingers >= min_force_fingers
            )
            if should_release:
                _set_object_joint_locked(model, data, False)
                released = True
                force_hit_step = step
                print(
                    f"Released object at step={step} "
                    f"contact_force={contact_force:.6g} contacts={contact_count} "
                    f"active_fingers={active_fingers}",
                    flush=True,
                )
            if step % 60 == 0:
                force_summary = " ".join(f"{name.split('_')[1][:2]}={value:.2f}" for name, value in fingertip_forces.items())
                print(
                    f"step={step} alpha={alpha:.3f} released={released} "
                    f"contact_force={contact_force:.6g} contacts={contact_count} "
                    f"active_fingers={active_fingers} {force_summary}",
                    flush=True,
                )
            viewer.sync()
            step += 1
            time.sleep(float(model.opt.timestep))
    if force_hit_step < 0 and release_after_force:
        print("Object was not released: contact force threshold was not reached.")


def _launch_manual_joint_slider_viewer(
    xml_path: Path,
    object_center: np.ndarray,
    joint_names: list[str],
    initial_values: np.ndarray,
    *,
    external_hand: bool,
    hold_initial_controls: bool,
    freeze_initial_qpos: bool,
    max_viewer_steps: int,
) -> None:
    import time

    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if external_hand:
        _set_external_hand_state(model, data, initial_values, object_center)
    else:
        _set_dynamic_state(model, data, joint_names, initial_values, initial_values.copy(), object_center)
    model.qpos0[:] = data.qpos
    _set_object_joint_locked(model, data, True)

    release_requested = False

    def key_callback(key: int) -> None:
        nonlocal release_requested
        if key in (ord("R"), ord("r")):
            release_requested = True

    print(
        "Manual viewer: use MuJoCo's right-side Control sliders for Shadow Hand joints. "
        "Press R in the viewer to release the object.",
        flush=True,
    )

    released = False
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback, show_right_ui=True) as viewer:
        step = 0
        while viewer.is_running() and (max_viewer_steps <= 0 or step < max_viewer_steps):
            with viewer.lock():
                if hold_initial_controls:
                    if external_hand:
                        for robot0_name, rh_name in ROBOT0_TO_RH_JOINT.items():
                            actuator_id = mujoco.mj_name2id(
                                model,
                                mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"rh_A_{rh_name.removeprefix('rh_')}",
                            )
                            if actuator_id >= 0:
                                data.ctrl[actuator_id] = _external_joint_value(robot0_name, initial_values)
                    else:
                        data.ctrl[:] = initial_values
                if freeze_initial_qpos:
                    if external_hand:
                        _set_external_hand_joints(model, data, initial_values)
                    else:
                        for joint_name, joint_value in zip(joint_names, initial_values):
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            if joint_id >= 0:
                                data.qpos[model.jnt_qposadr[joint_id]] = joint_value
                                data.qvel[model.jnt_dofadr[joint_id]] = 0.0
                if release_requested and not released:
                    _set_object_joint_locked(model, data, False)
                    released = True
                    print("Released object from manual viewer.", flush=True)
                if not released:
                    _reset_object_pose(model, data, object_center)
                if freeze_initial_qpos:
                    mujoco.mj_forward(model, data)
                else:
                    mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            time.sleep(float(model.opt.timestep))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and smoke-test a dynamic MuJoCo Shadow Hand scene from a GraspQP result.")
    parser.add_argument("--input", default="outputs/test_object_fingertips_20c.pt")
    parser.add_argument("--sample", type=int, default=2)
    parser.add_argument("--mesh-path", default=None)
    parser.add_argument("--out-dir", default="outputs/mujoco_dynamic_shadow_hand")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--friction", type=float, nargs=3, default=(5.0, 0.5, 0.05), metavar=("SLIDE", "TORSION", "ROLL"))
    parser.add_argument("--condim", type=int, default=4, choices=[1, 3, 4, 6])
    parser.add_argument("--object-mass", type=float, default=0.05)
    parser.add_argument("--fix-object", action="store_true", help="Keep the object fixed at its initial pose instead of adding a free joint.")
    parser.add_argument("--full-object-mesh", action="store_true")
    parser.add_argument("--contact-margin", type=float, default=0.0, help="MuJoCo geom contact margin for generated scenes.")
    parser.add_argument("--show-collision-geoms", action="store_true", help="Render hand collision geoms as translucent blue geoms.")
    parser.add_argument("--floor-z", type=float, default=-0.05)
    parser.add_argument("--floor-size", type=float, default=0.25)
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--actuator-kp", type=float, default=40.0)
    parser.add_argument("--actuator-force", type=float, default=5.0)
    parser.add_argument("--thumb-actuator-kp", type=float, default=None, help="Override MuJoCo thumb position actuator kp values.")
    parser.add_argument("--thumb-actuator-force", type=float, default=None, help="Override MuJoCo thumb position actuator force ranges to +/- this value.")
    parser.add_argument("--joint-damping", type=float, default=0.05)
    parser.add_argument("--preload-fraction", type=float, default=0.0, help="Move selected finger actuator targets toward closing limits by this fraction.")
    parser.add_argument("--preload-steps", type=int, default=0, help="Run this many hand-closing steps before release.")
    parser.add_argument("--free-object-during-preload", action="store_true", help="Let the object move during preload instead of holding it at the initial pose.")
    parser.add_argument("--gravity-during-preload", action="store_true", help="Keep gravity enabled during preload.")
    parser.add_argument("--hand-convex-collision", action="store_true", help="Export convex hulls for hand mesh geoms before loading in MuJoCo.")
    parser.add_argument("--full-contact-mesh", action="store_true", help="Use full Shadow Hand contact mesh OBJs as collision geoms on contact links.")
    parser.add_argument("--contact-sphere-radius", type=float, default=0.0, help="Attach small collision spheres at selected GraspQP contact candidates.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer with qpos and actuator ctrl initialized from the GraspQP result.")
    parser.add_argument("--ramp-release-viewer", action="store_true", help="Open a viewer that ramps from an open hand to the target pose while holding the object fixed, then releases it after a contact-force threshold.")
    parser.add_argument("--manual-viewer", action="store_true", help="Open MuJoCo viewer for manual grasping. Use native Control sliders and press R to release the object.")
    parser.add_argument(
        "--mujoco-hand-xml",
        default=str(LOCAL_MUJOCO_HAND_XML),
        help="MJCF hand model for --manual-viewer. Default uses the vendored MuJoCo Shadow Hand asset. Use 'graspqp' for the generated GraspQP hand.",
    )
    parser.add_argument(
        "--manual-object-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional object position override used with --manual-viewer and --mujoco-hand-xml. Default keeps the GraspQP object position.",
    )
    parser.add_argument(
        "--manual-initial-pose",
        choices=["default", "opened", "optimized"],
        default="optimized",
        help="Initial hand pose for --manual-viewer. Use 'opened' to keep the GraspQP wrist/root pose and open only the fingers.",
    )
    parser.add_argument(
        "--hold-initial-controls",
        action="store_true",
        help="In --manual-viewer, keep actuator controls fixed at the selected initial pose instead of using the native control sliders.",
    )
    parser.add_argument(
        "--freeze-initial-qpos",
        action="store_true",
        help="In --manual-viewer, force the hand qpos to the selected initial pose every step for static pose inspection.",
    )
    parser.add_argument("--open-fraction", type=float, default=0.7, help="How far to open closing joints away from the target before ramping.")
    parser.add_argument("--ramp-steps", type=int, default=300)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--release-force-threshold", type=float, default=2.0)
    parser.add_argument("--per-finger-force-threshold", type=float, default=0.2)
    parser.add_argument("--min-force-fingers", type=int, default=2)
    parser.add_argument("--close-until-force", action="store_true", help="Close toward the target pose until enough fingertips reach the per-finger force threshold.")
    parser.add_argument("--stop-fingers-on-force", action="store_true", help="With --close-until-force, hold each finger once its distal link exceeds --per-finger-force-threshold.")
    parser.add_argument("--close-rate", type=float, default=0.002, help="Joint target increment per simulation step for --close-until-force.")
    parser.add_argument("--thumb-thj4-offset", type=float, default=0.0, help="Add a small offset to robot0_THJ4 before building the MuJoCo scene. Positive values lift the thumb in world z for the current sample.")
    parser.add_argument("--no-release-after-force", action="store_true", help="Ramp and hold object fixed without releasing it.")
    parser.add_argument("--max-viewer-steps", type=int, default=0, help="Stop ramp-release viewer after N steps. Default 0 runs until closed.")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    sample = _select_sample(payload, args.sample)
    joint_values = payload["final_state"]["joint_values"][sample].detach().cpu().numpy().astype(np.float64)
    urdf_root, children_by_parent, _, _ = _parse_urdf(resolve_shadow_hand_asset_dir() / "shadow_hand.urdf")
    del urdf_root
    limits = _joint_limits(children_by_parent)
    _apply_joint_offset(joint_values, "robot0_THJ4", args.thumb_thj4_offset, limits)
    ctrl_values = _preload_targets(joint_values, limits, args.preload_fraction)
    open_values = _open_hand_targets(joint_values, limits, args.open_fraction)
    manual_initial_values = {
        "default": DEFAULT_JOINT_STATE.detach().cpu().numpy().astype(np.float64),
        "opened": open_values,
        "optimized": joint_values,
    }[args.manual_initial_pose]
    using_external_hand = args.mujoco_hand_xml != "graspqp" and not args.ramp_release_viewer and not args.viewer
    if using_external_hand:
        xml_path, object_center, joint_names = _build_external_mujoco_hand_scene(
            payload,
            sample,
            Path(args.out_dir),
            hand_xml=args.mujoco_hand_xml,
            mesh_path=args.mesh_path,
            hand_joint_values=manual_initial_values if args.manual_viewer else joint_values,
            object_position=None
            if args.manual_object_position is None
            else tuple(float(x) for x in args.manual_object_position),
            timestep=args.timestep,
            gravity=(0.0, 0.0, args.gravity_z),
            friction=tuple(float(x) for x in args.friction),
            condim=args.condim,
            object_mass=args.object_mass,
            object_convex_hull=not args.full_object_mesh,
            floor_z=None if args.no_floor else args.floor_z,
            floor_size=args.floor_size,
            contact_margin=args.contact_margin,
            show_collision_geoms=args.show_collision_geoms,
            thumb_actuator_kp=args.thumb_actuator_kp,
            thumb_actuator_force=args.thumb_actuator_force,
        )
    else:
        xml_path, object_center, joint_names = _build_dynamic_scene(
            payload,
            sample,
            Path(args.out_dir),
            mesh_path=args.mesh_path,
            timestep=args.timestep,
            gravity=(0.0, 0.0, args.gravity_z),
            friction=tuple(float(x) for x in args.friction),
            condim=args.condim,
            object_mass=args.object_mass,
            object_convex_hull=not args.full_object_mesh,
            fix_object=args.fix_object and not args.ramp_release_viewer and not args.manual_viewer,
            floor_z=None if args.no_floor else args.floor_z,
            floor_size=args.floor_size,
            actuator_kp=args.actuator_kp,
            actuator_force=args.actuator_force,
            thumb_actuator_kp=args.thumb_actuator_kp,
            thumb_actuator_force=args.thumb_actuator_force,
            joint_damping=args.joint_damping,
            hand_convex_collision=args.hand_convex_collision,
            full_contact_mesh=args.full_contact_mesh,
            contact_sphere_radius=args.contact_sphere_radius,
        )
    print(f"MuJoCo dynamic scene: {xml_path}")
    print(
        f"sample={sample} steps={args.steps} condim={args.condim} friction={tuple(args.friction)} "
        f"preload_fraction={args.preload_fraction} hand_convex_collision={args.hand_convex_collision} "
        f"full_contact_mesh={args.full_contact_mesh}"
    )
    if args.manual_viewer:
        print(
            f"manual_initial_pose={args.manual_initial_pose} "
            f"hold_initial_controls={args.hold_initial_controls} freeze_initial_qpos={args.freeze_initial_qpos}"
        )
    if not args.manual_viewer:
        if using_external_hand:
            metrics = _run_external_dynamic_smoke(
                xml_path,
                object_center,
                joint_values,
                args.steps,
                preload_steps=args.preload_steps,
                freeze_object_during_preload=not args.free_object_during_preload,
                zero_gravity_during_preload=not args.gravity_during_preload,
            )
        else:
            metrics = _run_dynamic_smoke(
                xml_path,
                object_center,
                joint_names,
                joint_values,
                ctrl_values,
                args.steps,
                preload_steps=args.preload_steps,
                freeze_object_during_preload=not args.free_object_during_preload,
                zero_gravity_during_preload=not args.gravity_during_preload,
            )
        for key, value in metrics.items():
            print(f"{key}: {value:.9g}")
    if args.manual_viewer:
        _launch_manual_joint_slider_viewer(
            xml_path,
            object_center,
            joint_names,
            manual_initial_values,
            external_hand=using_external_hand,
            hold_initial_controls=args.hold_initial_controls,
            freeze_initial_qpos=args.freeze_initial_qpos,
            max_viewer_steps=args.max_viewer_steps,
        )
    elif args.ramp_release_viewer:
        _launch_ramp_release_viewer(
            xml_path,
            object_center,
            joint_names,
            open_values,
            ctrl_values,
            ramp_steps=args.ramp_steps,
            hold_steps=args.hold_steps,
            force_threshold=args.release_force_threshold,
            release_after_force=not args.no_release_after_force,
            per_finger_force_threshold=args.per_finger_force_threshold,
            min_force_fingers=args.min_force_fingers,
            close_until_force=args.close_until_force,
            stop_fingers_on_force=args.stop_fingers_on_force,
            close_rate=args.close_rate,
            max_viewer_steps=args.max_viewer_steps,
        )
    elif args.viewer:
        _launch_dynamic_viewer(
            xml_path,
            object_center,
            joint_names,
            joint_values,
            ctrl_values,
            preload_steps=args.preload_steps,
            freeze_object_during_preload=not args.free_object_during_preload,
            zero_gravity_during_preload=not args.gravity_during_preload,
        )


if __name__ == "__main__":
    main()
