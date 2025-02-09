# pylint: disable=E1136
# pylint: disable=E1137
from typing import Dict, List, Union, Optional
from collections import OrderedDict
from pydantic import BaseModel, Field

from ase import Atoms
from ase.build.tools import sort

import torch
from torch import Tensor


class TrajectoryStep(BaseModel):
    num_atoms: Tensor
    atom_types: Tensor
    frac_coords: Tensor
    lattices: Tensor
    batch_idx: Tensor

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class TrajectoryContainer(BaseModel):
    """
    key is the time step, value is a list of TrajectoryStep
    """

    trajectory_continaer: Dict[int, List[TrajectoryStep]] = Field(
        default_factory=OrderedDict
    )
    total_steps: int

    def __init__(self, total_steps: int):
        super().__init__(total_steps=total_steps)
        self.trajectory_continaer = OrderedDict()
        self.total_steps = total_steps
        for t in range(self.total_steps):
            self.trajectory_continaer[t] = None

    def __setitem__(self, t: int, step: TrajectoryStep) -> None:
        self.trajectory_continaer[t] = step

    def __getitem__(self, t: int) -> TrajectoryStep:
        if t == -1:
            t = self.total_steps
        return self.trajectory_continaer[t]

    def __len__(self) -> int:
        return len(self.trajectory_continaer)

    def __iter__(self):
        return iter(self.trajectory_continaer)

    def get_atoms(self, t: int = 0, idx: int = None) -> Union[Atoms, List[Atoms]]:
        trajectory_step = self[t]
        # if atom type is greater than 103, set it to 0
        trajectory_step.atom_types = torch.where(
            trajectory_step.atom_types <= 103, trajectory_step.atom_types, 0
        )
        split_atom_types = torch.split(
            trajectory_step.atom_types, trajectory_step.num_atoms.tolist()
        )
        split_frac_coords = torch.split(
            trajectory_step.frac_coords, trajectory_step.num_atoms.tolist()
        )
        atoms_list = []
        for i, (frac_coords, atom_types) in enumerate(
            zip(split_frac_coords, split_atom_types)
        ):
            atoms = Atoms(
                numbers=atom_types.detach().cpu().numpy(),
                cell=trajectory_step.lattices[i].detach().cpu().numpy(),
                pbc=True,
            )
            positions = frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
            atoms_list.append(sort(atoms))
        if idx is None:
            return atoms_list
        return atoms_list[idx]

    def get_trajectory(self, idx: int = None) -> Union[List[Atoms], List[List[Atoms]]]:
        if idx is None:
            return [self.get_atoms(t) for t in range(self.total_steps + 1)]
        return [self.get_atoms(t, idx) for t in range(self.total_steps + 1)]
