from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraspState:
    joint_values: torch.Tensor
    wrist_translation: torch.Tensor
    wrist_rotation: torch.Tensor
    contact_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.joint_values.shape[0])

    def clone(self) -> "GraspState":
        return GraspState(
            joint_values=self.joint_values.clone(),
            wrist_translation=self.wrist_translation.clone(),
            wrist_rotation=self.wrist_rotation.clone(),
            contact_indices=self.contact_indices.clone(),
        )

    def detached(self) -> "GraspState":
        return GraspState(
            joint_values=self.joint_values.detach(),
            wrist_translation=self.wrist_translation.detach(),
            wrist_rotation=self.wrist_rotation.detach(),
            contact_indices=self.contact_indices.detach(),
        )
