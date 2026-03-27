from __future__ import annotations

import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox, ttk

import torch

from minimal_graspqp.energy import compute_grasp_energy
from minimal_graspqp.hands import ShadowHandModel
from minimal_graspqp.init import initialize_grasps_for_primitive
from minimal_graspqp.metrics import ForceClosureQP
from minimal_graspqp.objects import Box, Cylinder, MeshObject, Sphere
from minimal_graspqp.optim import MalaConfig, MalaOptimizer
from minimal_graspqp.rotation import palm_down_rotation
from minimal_graspqp.visualization import (
    publish_initialization_viser,
    publish_optimization_result_viser,
    publish_shadow_hand_primitive_viser,
)


def build_object(config: dict):
    object_mode = config["object_mode"]
    if object_mode == "mesh":
        return MeshObject(config["mesh_path"], scale=config["mesh_scale"])
    if object_mode == "object_code":
        return MeshObject.from_code(config["object_root"], config["object_code"], scale=config["mesh_scale"])
    if object_mode == "sphere":
        return Sphere(radius=config["radius"])
    if object_mode == "cylinder":
        return Cylinder(radius=config["radius"], half_height=config["half_height"])
    if object_mode == "box":
        return Box(half_extents=(config["half_x"], config["half_y"], config["half_z"]))
    raise ValueError(f"Unsupported object mode: {object_mode}")


class GraspUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("minimal_graspqp UI")
        self._worker = None

        self.object_mode = StringVar(value="object_code")
        self.mesh_path = StringVar(value="/home/haegu/minimal_graspqp/assets/objects/test_object.stl")
        self.object_root = StringVar(value="/home/haegu/minimal_graspqp/assets/objects")
        self.object_code = StringVar(value="core_bottle")
        self.mesh_scale = DoubleVar(value=1.0)
        self.primitive = StringVar(value="sphere")
        self.radius = DoubleVar(value=0.05)
        self.half_height = DoubleVar(value=0.08)
        self.half_x = DoubleVar(value=0.04)
        self.half_y = DoubleVar(value=0.04)
        self.half_z = DoubleVar(value=0.04)
        self.device = StringVar(value="cpu")
        self.batch_size = IntVar(value=4)
        self.num_contacts = IntVar(value=12)
        self.num_steps = IntVar(value=200)
        self.seed = IntVar(value=0)
        self.step_size = DoubleVar(value=5e-3)
        self.temperature = DoubleVar(value=18.0)
        self.temperature_decay = DoubleVar(value=0.95)
        self.contact_switch_probability = DoubleVar(value=0.4)
        self.output_pt = StringVar(value="outputs/ui_optimization.pt")
        self.output_html = StringVar(value="outputs/ui_optimization.html")
        self.palm_down = BooleanVar(value=False)
        self.mala_star = BooleanVar(value=True)
        self.fingertips_only = BooleanVar(value=False)
        self.distance_lower = DoubleVar(value=0.08)
        self.distance_upper = DoubleVar(value=0.12)
        self.sample_index = IntVar(value=0)
        self.status = StringVar(value="Idle")
        self.viewer_port = IntVar(value=8080)
        self._viewer_server = None

        self._build()

    def _build(self):
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0

        def add_entry(label: str, variable, width: int = 44):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(frame, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", pady=2)
            row += 1

        def add_check(label: str, variable):
            nonlocal row
            ttk.Checkbutton(frame, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            row += 1

        def add_combo(label: str, variable, values):
            nonlocal row
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Combobox(frame, textvariable=variable, values=list(values), state="readonly", width=20).grid(
                row=row, column=1, sticky="w", pady=2
            )
            row += 1

        frame.columnconfigure(1, weight=1)

        add_combo("Object Mode", self.object_mode, ["object_code", "mesh", "sphere", "cylinder", "box"])
        add_entry("Mesh Path", self.mesh_path)
        ttk.Button(frame, text="Browse", command=self._browse_mesh).grid(row=row - 1, column=2, padx=(8, 0), sticky="w")
        add_entry("Object Root", self.object_root)
        add_entry("Object Code", self.object_code)
        add_entry("Mesh Scale", self.mesh_scale)
        add_combo("Primitive", self.primitive, ["sphere", "cylinder", "box"])
        add_entry("Radius", self.radius)
        add_entry("Half Height", self.half_height)
        add_entry("Half X", self.half_x)
        add_entry("Half Y", self.half_y)
        add_entry("Half Z", self.half_z)
        add_entry("Device", self.device)
        add_entry("Batch Size", self.batch_size)
        add_entry("Sample Index", self.sample_index)
        add_entry("Num Contacts", self.num_contacts)
        add_entry("Num Steps", self.num_steps)
        add_entry("Seed", self.seed)
        add_entry("Distance Lower", self.distance_lower)
        add_entry("Distance Upper", self.distance_upper)
        add_entry("Step Size", self.step_size)
        add_entry("Temperature", self.temperature)
        add_entry("Temp Decay", self.temperature_decay)
        add_entry("Contact Switch Prob", self.contact_switch_probability)
        add_entry("Output PT", self.output_pt)
        add_entry("Output HTML", self.output_html)
        add_entry("Viewer Port", self.viewer_port)
        add_check("Palm Down", self.palm_down)
        add_check("Use MALA*", self.mala_star)
        add_check("Fingertips Only", self.fingertips_only)

        button_row = ttk.Frame(frame)
        button_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 6))
        ttk.Button(button_row, text="Preview Hand", command=self.preview_hand).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Preview Init", command=self.preview_initialization).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Run Optimize", command=self.run_optimization).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Open Result HTML", command=self.open_result_html).pack(side="left")
        row += 1

        ttk.Label(frame, textvariable=self.status).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _browse_mesh(self):
        path = filedialog.askopenfilename(
            title="Select mesh file",
            filetypes=[("Mesh files", "*.obj *.stl *.ply"), ("All files", "*.*")],
        )
        if path:
            self.mesh_path.set(path)
            self.object_mode.set("mesh")

    def _config(self) -> dict:
        return {
            "object_mode": self.object_mode.get(),
            "mesh_path": self.mesh_path.get(),
            "object_root": self.object_root.get(),
            "object_code": self.object_code.get(),
            "mesh_scale": float(self.mesh_scale.get()),
            "primitive": self.primitive.get(),
            "radius": float(self.radius.get()),
            "half_height": float(self.half_height.get()),
            "half_x": float(self.half_x.get()),
            "half_y": float(self.half_y.get()),
            "half_z": float(self.half_z.get()),
            "device": self.device.get(),
            "batch_size": int(self.batch_size.get()),
            "sample_index": int(self.sample_index.get()),
            "num_contacts": int(self.num_contacts.get()),
            "num_steps": int(self.num_steps.get()),
            "seed": int(self.seed.get()),
            "distance_lower": float(self.distance_lower.get()),
            "distance_upper": float(self.distance_upper.get()),
            "step_size": float(self.step_size.get()),
            "temperature": float(self.temperature.get()),
            "temperature_decay": float(self.temperature_decay.get()),
            "contact_switch_probability": float(self.contact_switch_probability.get()),
            "output_pt": self.output_pt.get(),
            "output_html": self.output_html.get(),
            "viewer_port": int(self.viewer_port.get()),
            "palm_down": bool(self.palm_down.get()),
            "mala_star": bool(self.mala_star.get()),
            "fingertips_only": bool(self.fingertips_only.get()),
        }

    def _set_status(self, text: str):
        self.status.set(text)
        self.root.update_idletasks()

    def _start_worker(self, fn):
        if self._worker is not None and self._worker.is_alive():
            messagebox.showinfo("Busy", "A job is already running.")
            return
        self._worker = threading.Thread(target=fn, daemon=True)
        self._worker.start()

    def _preview_output_path(self, suffix: str) -> Path:
        output_path = Path(self.output_html.get())
        return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")

    def _replace_viewer(self, server, port: int):
        if self._viewer_server is not None:
            self._viewer_server.stop()
        self._viewer_server = server
        webbrowser.open(f"http://localhost:{port}")

    def preview_hand(self):
        config = self._config()

        def task():
            try:
                self._set_status("Building hand anatomy preview...")
                hand_model = ShadowHandModel.create(device=config["device"], fingertips_only=config["fingertips_only"])
                obj = build_object(config)
                joint_values = hand_model.default_joint_state(batch_size=1)
                wrist_translation = torch.zeros((1, 3), device=hand_model.device, dtype=hand_model.dtype)
                wrist_rotation = torch.eye(3, device=hand_model.device, dtype=hand_model.dtype).unsqueeze(0)
                if config["palm_down"]:
                    wrist_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device).unsqueeze(0)
                server = publish_shadow_hand_primitive_viser(
                    hand_model,
                    obj,
                    joint_values,
                    wrist_translation=wrist_translation,
                    wrist_rotation=wrist_rotation,
                    show_penetration_spheres=True,
                    host="0.0.0.0",
                    port=config["viewer_port"],
                )
                self.root.after(0, lambda: self._replace_viewer(server, config["viewer_port"]))
                self._set_status(f"Hand anatomy preview running on http://localhost:{config['viewer_port']}")
            except Exception as exc:
                self._set_status("Hand anatomy preview failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"{exc}\n\n{traceback.format_exc()}"))

        self._start_worker(task)

    def preview_initialization(self):
        config = self._config()

        def task():
            try:
                self._set_status("Building initialization preview...")
                torch.manual_seed(config["seed"])
                hand_model = ShadowHandModel.create(device=config["device"], fingertips_only=config["fingertips_only"])
                obj = build_object(config)
                base_rotation = None
                if config["palm_down"]:
                    base_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
                state = initialize_grasps_for_primitive(
                    hand_model,
                    obj,
                    batch_size=config["batch_size"],
                    distance_lower=config["distance_lower"],
                    distance_upper=config["distance_upper"],
                    num_contacts=config["num_contacts"],
                    base_wrist_rotation=base_rotation,
                )
                server = publish_initialization_viser(
                    hand_model,
                    obj,
                    state,
                    host="0.0.0.0",
                    port=config["viewer_port"],
                )
                self.root.after(0, lambda: self._replace_viewer(server, config["viewer_port"]))
                self._set_status(f"Initialization preview running on http://localhost:{config['viewer_port']}")
            except Exception as exc:
                self._set_status("Initialization preview failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"{exc}\n\n{traceback.format_exc()}"))

        self._start_worker(task)

    def run_optimization(self):
        config = self._config()

        def task():
            try:
                self._set_status("Running optimization...")
                torch.manual_seed(config["seed"])
                hand_model = ShadowHandModel.create(device=config["device"], fingertips_only=config["fingertips_only"])
                obj = build_object(config)
                base_rotation = None
                if config["palm_down"]:
                    base_rotation = palm_down_rotation(dtype=hand_model.dtype, device=hand_model.device)
                initial_state = initialize_grasps_for_primitive(
                    hand_model,
                    obj,
                    batch_size=config["batch_size"],
                    distance_lower=config["distance_lower"],
                    distance_upper=config["distance_upper"],
                    num_contacts=config["num_contacts"],
                    base_wrist_rotation=base_rotation,
                )
                metric = ForceClosureQP(min_force=0.0, max_force=20.0)
                optimizer = MalaOptimizer(
                    MalaConfig(
                        num_steps=config["num_steps"],
                        step_size=config["step_size"],
                        temperature=config["temperature"],
                        temperature_decay=config["temperature_decay"],
                        contact_switch_probability=config["contact_switch_probability"],
                        use_mala_star=config["mala_star"],
                    )
                )
                initial_losses = compute_grasp_energy(hand_model, obj, initial_state, metric)
                final_state, history = optimizer.optimize(hand_model, obj, initial_state, metric)
                final_losses = compute_grasp_energy(hand_model, obj, final_state, metric)

                output_pt = Path(config["output_pt"])
                output_pt.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "primitive": {
                        "type": "mesh" if isinstance(obj, MeshObject) else config["primitive"],
                        "mesh_path": str(getattr(obj, "mesh_path", "")),
                        "scale": float(config["mesh_scale"]),
                        "center": list(obj.center),
                    },
                    "hand": {"fingertips_only": bool(config["fingertips_only"])},
                    "initial_state": {
                        "joint_values": initial_state.joint_values.detach().cpu(),
                        "wrist_translation": initial_state.wrist_translation.detach().cpu(),
                        "wrist_rotation": initial_state.wrist_rotation.detach().cpu(),
                        "contact_indices": initial_state.contact_indices.detach().cpu(),
                    },
                    "final_state": {
                        "joint_values": final_state.joint_values.detach().cpu(),
                        "wrist_translation": final_state.wrist_translation.detach().cpu(),
                        "wrist_rotation": final_state.wrist_rotation.detach().cpu(),
                        "contact_indices": final_state.contact_indices.detach().cpu(),
                    },
                    "initial_energy": initial_losses["E_total"].detach().cpu(),
                    "final_energy": final_losses["E_total"].detach().cpu(),
                    "energy_trace": torch.stack(history.energy_trace).cpu(),
                }
                torch.save(payload, output_pt)

                sample_index = max(0, min(config["sample_index"], config["batch_size"] - 1))
                server = publish_optimization_result_viser(
                    hand_model,
                    obj,
                    initial_state=initial_state,
                    final_state=final_state,
                    sample_index=sample_index,
                    host="0.0.0.0",
                    port=config["viewer_port"],
                )
                self.root.after(0, lambda: self._replace_viewer(server, config["viewer_port"]))
                self._set_status(
                    f"Done. mean initial={initial_losses['E_total'].mean().item():.3f}, "
                    f"mean final={final_losses['E_total'].mean().item():.3f}. "
                    f"Viewer: http://localhost:{config['viewer_port']}"
                )
            except Exception as exc:
                self._set_status("Optimization failed")
                self.root.after(0, lambda: messagebox.showerror("Error", f"{exc}\n\n{traceback.format_exc()}"))

        self._start_worker(task)

    def open_result_html(self):
        webbrowser.open(f"http://localhost:{self.viewer_port.get()}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    GraspUI().run()
