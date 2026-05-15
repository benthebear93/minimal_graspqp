from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh as tm
from scipy.spatial.transform import Rotation

from minimal_graspqp.hands import ShadowHandModel


def _as_float_list(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def _quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return vertices @ rotation.T + translation.reshape(1, 3)


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh = tm.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.export(path)


def _state_tensor(state: dict[str, torch.Tensor], name: str, sample: int) -> torch.Tensor:
    value = state[name]
    return value[sample : sample + 1].detach().cpu()


def _select_sample(payload: dict, sample: int | None) -> int:
    final_energy = payload.get("final_energy")
    if sample is not None:
        return int(sample)
    if isinstance(final_energy, torch.Tensor):
        return int(torch.argmin(final_energy).item())
    return 0


def _load_object_mesh(primitive: dict, fallback_mesh: str | None, convex_hull: bool) -> tuple[tm.Trimesh, np.ndarray]:
    if primitive.get("type") != "mesh" and fallback_mesh is None:
        raise ValueError("MuJoCo smoke test currently expects a mesh result or --mesh-path.")
    mesh_path = Path(fallback_mesh or primitive["mesh_path"]).expanduser().resolve()
    mesh = tm.load(mesh_path, force="mesh", process=True)
    scale = float(primitive.get("scale", 1.0))
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    rpy = primitive.get("rotation_rpy") or [0.0, 0.0, 0.0]
    if any(abs(float(angle)) > 1e-12 for angle in rpy):
        rotation = _rotation_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
    center = np.asarray(mesh.centroid, dtype=np.float64)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) - center.reshape(1, 3)
    if convex_hull:
        mesh = mesh.convex_hull
    return mesh, center


def _load_object_visual_collision_meshes(
    primitive: dict,
    fallback_mesh: str | None,
    convex_hull: bool,
) -> tuple[tm.Trimesh, tm.Trimesh, np.ndarray]:
    visual_mesh, center = _load_object_mesh(primitive, fallback_mesh, convex_hull=False)
    if convex_hull:
        collision_mesh = visual_mesh.convex_hull
    else:
        collision_mesh = visual_mesh.copy()
    return visual_mesh, collision_mesh, center


def _add_hand_mesh_geoms(
    asset: ET.Element,
    worldbody: ET.Element,
    hand_model: ShadowHandModel,
    joint_values: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotation: torch.Tensor,
    mesh_dir: Path,
    friction: tuple[float, float, float],
    condim: int,
) -> int:
    transforms = hand_model.forward_kinematics(joint_values.to(device=hand_model.device, dtype=hand_model.dtype))
    wrist_r = wrist_rotation.squeeze(0).detach().cpu().numpy()
    wrist_t = wrist_translation.squeeze(0).detach().cpu().numpy()

    count = 0
    for link_name, mesh_data in sorted(hand_model._collision_meshes.items()):
        link_transform = transforms[link_name][0].detach().cpu().numpy()
        link_r = link_transform[:3, :3]
        link_t = link_transform[:3, 3]
        world_r = wrist_r @ link_r
        world_t = wrist_r @ link_t + wrist_t

        if "vertices" in mesh_data and "faces" in mesh_data:
            vertices = mesh_data["vertices"].detach().cpu().numpy()
            faces = mesh_data["faces"].detach().cpu().numpy()
            world_vertices = _transform_vertices(vertices, world_r, world_t)
            mesh_name = f"hand_{link_name}"
            mesh_path = mesh_dir / f"{mesh_name}.obj"
            _write_mesh(mesh_path, world_vertices, faces)
            ET.SubElement(asset, "mesh", name=mesh_name, file=str(mesh_path.resolve()))
            ET.SubElement(
                worldbody,
                "geom",
                name=f"{mesh_name}_geom",
                type="mesh",
                mesh=mesh_name,
                friction=_as_float_list(friction),
                condim=str(condim),
            )
            count += 1

        for primitive_id, primitive in enumerate(mesh_data.get("primitives", [])):
            prim_r = primitive["rotation"].detach().cpu().numpy()
            prim_t = primitive["translation"].detach().cpu().numpy()
            geom_r = world_r @ prim_r
            geom_t = world_r @ prim_t + world_t
            attrs = {
                "name": f"hand_{link_name}_primitive_{primitive_id}",
                "pos": _as_float_list(geom_t),
                "quat": _as_float_list(_quat_wxyz(geom_r)),
                "friction": _as_float_list(friction),
                "condim": str(condim),
            }
            if primitive["type"] == "box":
                attrs["type"] = "box"
                attrs["size"] = _as_float_list(primitive["half_extents"].detach().cpu().numpy())
            elif primitive["type"] == "sphere":
                attrs["type"] = "sphere"
                attrs["size"] = f"{float(primitive['radius']):.9g}"
            elif primitive["type"] == "cylinder":
                attrs["type"] = "cylinder"
                attrs["size"] = f"{float(primitive['radius']):.9g} {float(primitive['half_height']):.9g}"
            else:
                raise ValueError(f"Unsupported primitive collision type: {primitive['type']}")
            ET.SubElement(worldbody, "geom", **attrs)
            count += 1
    return count


def _build_scene(
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
    floor_z: float | None,
    floor_size: float,
) -> tuple[Path, np.ndarray, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    hand_info = payload.get("hand", {})
    hand_model = ShadowHandModel.create(
        device="cpu",
        fingertips_only=bool(hand_info.get("fingertips_only", False)),
        allowed_contact_links=hand_info.get("allowed_contact_links"),
    )
    final_state = payload["final_state"]
    joint_values = _state_tensor(final_state, "joint_values", sample)
    wrist_translation = _state_tensor(final_state, "wrist_translation", sample)
    wrist_rotation = _state_tensor(final_state, "wrist_rotation", sample)

    root = ET.Element("mujoco", model="minimal_graspqp_static_hand_grasp")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=f"{timestep:.9g}", gravity=_as_float_list(gravity), integrator="implicitfast")
    ET.SubElement(root, "size", njmax="2000", nconmax="500")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", solref="0.01 1", solimp="0.9 0.95 0.001", margin="0.0005")
    asset = ET.SubElement(root, "asset")
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

    hand_geom_count = _add_hand_mesh_geoms(
        asset,
        worldbody,
        hand_model,
        joint_values,
        wrist_translation,
        wrist_rotation,
        mesh_dir,
        friction,
        condim,
    )

    object_mesh, object_center = _load_object_mesh(payload["primitive"], mesh_path, convex_hull=object_convex_hull)
    object_mesh_path = mesh_dir / "object.obj"
    _write_mesh(object_mesh_path, np.asarray(object_mesh.vertices), np.asarray(object_mesh.faces))
    ET.SubElement(asset, "mesh", name="object_mesh", file=str(object_mesh_path.resolve()))
    object_body = ET.SubElement(worldbody, "body", name="object", pos=_as_float_list(object_center))
    ET.SubElement(object_body, "freejoint", name="object_free")
    ET.SubElement(
        object_body,
        "geom",
        name="object_geom",
        type="mesh",
        mesh="object_mesh",
        mass=f"{object_mass:.9g}",
        friction=_as_float_list(friction),
        condim=str(condim),
    )

    xml_path = out_dir / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path, object_center, hand_geom_count


def _run_simulation(xml_path: Path, object_center: np.ndarray, steps: int) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 3] = object_center
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    initial_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")].copy()
    contact_counts = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        object_contacts = 0
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            if name1 == "object_geom" or name2 == "object_geom":
                object_contacts += 1
        contact_counts.append(object_contacts)

    final_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")].copy()
    displacement = final_pos - initial_pos
    return {
        "initial_x": float(initial_pos[0]),
        "initial_y": float(initial_pos[1]),
        "initial_z": float(initial_pos[2]),
        "final_x": float(final_pos[0]),
        "final_y": float(final_pos[1]),
        "final_z": float(final_pos[2]),
        "displacement_norm": float(np.linalg.norm(displacement)),
        "z_drop": float(initial_pos[2] - final_pos[2]),
        "mean_object_contacts": float(np.mean(contact_counts) if contact_counts else 0.0),
        "final_object_contacts": float(contact_counts[-1] if contact_counts else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a saved GraspQP result in MuJoCo with a fixed Shadow Hand.")
    parser.add_argument("--input", default="outputs/test_object_fingertips_20c.pt")
    parser.add_argument("--sample", type=int, default=None, help="Batch sample to simulate. Default chooses the lowest final energy.")
    parser.add_argument("--mesh-path", default=None, help="Optional mesh path override for legacy result files.")
    parser.add_argument("--out-dir", default="outputs/mujoco_grasp_smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--friction", type=float, nargs=3, default=(1.2, 0.02, 0.001), metavar=("SLIDE", "TORSION", "ROLL"))
    parser.add_argument("--condim", type=int, default=4, choices=[1, 3, 4, 6])
    parser.add_argument("--object-mass", type=float, default=0.05)
    parser.add_argument("--full-object-mesh", action="store_true", help="Use the full object mesh instead of its convex hull.")
    parser.add_argument("--floor-z", type=float, default=-0.05, help="Add a floor plane at this z height. Use --no-floor to disable.")
    parser.add_argument("--floor-size", type=float, default=0.25, help="Half-size of the floor plane in meters.")
    parser.add_argument("--no-floor", action="store_true")
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    sample = _select_sample(payload, args.sample)
    xml_path, object_center, hand_geom_count = _build_scene(
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
        floor_z=None if args.no_floor else args.floor_z,
        floor_size=args.floor_size,
    )
    metrics = _run_simulation(xml_path, object_center, args.steps)
    print(f"MuJoCo scene: {xml_path}")
    print(f"sample={sample} hand_geoms={hand_geom_count} steps={args.steps} condim={args.condim} friction={tuple(args.friction)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.9g}")


if __name__ == "__main__":
    main()
