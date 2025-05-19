import sys
from pathlib import Path
from collections import Counter
import argparse
import os
import json
import concurrent.futures


from tqdm import tqdm
from p_tqdm import p_map
import pandas as pd
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty
import itertools
import numpy as np

import smact
from smact.screening import pauling_test


chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
    assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
        cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


sys.path.append(".")

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

Percentiles = {
    "mp20": np.array([-3.17562208, -2.82196882, -2.52814761]),
    "carbon": np.array([-154.527093, -154.45865733, -154.44206825]),
    "perovskite": np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.0},
    "carbon": {"struc": 0.2, "comp": 4.0},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]
        self.lengths = crys_array_dict["lengths"]
        self.angles = crys_array_dict["angles"]
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            self.dict["atom_types"] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1

        self.get_structure()
        self.get_composition()
        self.get_validity()
        # self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.stol = stol
        self.angle_tol = angle_tol
        self.ltol = ltol
        self.preds = pred_crys
        self.gts = gt_crys

    @staticmethod
    def process_one(args):
        pred, gt, is_valid, stol, angle_tol, ltol = args
        """Static method so it can be cleanly pickled by ProcessPoolExecutor."""
        # compositon matching
        comp_match = pred.structure.composition == gt.structure.composition

        if not is_valid:
            return None, comp_match

        from pymatgen.analysis.structure_matcher import (
            StructureMatcher,
        )  # must import inside method if we use multiple processes

        matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        try:
            rms_dist = matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist, comp_match
        except Exception:
            return None, comp_match

    def get_match_rate_and_rms(self):
        # Mark valid pairs
        validity = [c1.valid and c2.valid for c1, c2 in zip(self.preds, self.gts)]

        # Package all data into tuples for pickling
        data_to_process = [
            (pred, gt, valid, self.stol, self.angle_tol, self.ltol)
            for pred, gt, valid in zip(self.preds, self.gts, validity)
        ]

        # Parallel execution with a fixed number of 32 workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            # Optionally specify `chunksize` to reduce overhead, especially if len(self.preds) is large
            results_iter = executor.map(self.process_one, data_to_process)
            # Or if you want to control chunk size:
            # results_iter = executor.map(self.process_one, data_to_process, chunksize=50)

            # Collect results in order, with a progress bar
            results = list(
                tqdm(
                    results_iter,
                    total=len(data_to_process),
                    desc="Computing RMS distances (ProcessPool)",
                )
            )

        # Convert to numpy array so we can do numeric operations
        rms_dists = np.array(results, dtype=object)
        # Calculate match rate
        match_rate = np.count_nonzero(rms_dists != None) / len(self.preds)
        # Calculate mean RMS among all non-None results
        mean_rms_dist = (
            rms_dists[rms_dists != None].mean()
            if np.any(rms_dists != None)
            else float("nan")
        )

        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif, fmt="cif")
    lattice = structure.lattice
    crys_array_dict = {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(lattice.abc),
        "angles": np.array(lattice.angles),
    }
    return Crystal(crys_array_dict)


def main(args):
    all_metrics = {}

    # cfg = load_config(args.root_path)
    # eval_model_name = cfg.data.eval_model_name

    df_results = pd.read_csv(args.result_path)
    print(len(df_results))
    ref_crys_list = []
    gen_crys_list = []
    for _, row in tqdm(df_results.iterrows(), total=len(df_results)):
        try:
            gt_crys = get_gt_crys_ori(row["ref_structure"])
            for i in range(20):
                pred_crys = get_gt_crys_ori(row[f"gen_structure_{i}"])
                ref_crys_list.append(gt_crys)
                gen_crys_list.append(pred_crys)
        except Exception as e:
            print(e)
            continue
    print(f"Number of reference crystals: {len(ref_crys_list)}")
    print(f"Number of generated crystals: {len(gen_crys_list)}")

    rec_evaluator = RecEval(pred_crys=gen_crys_list, gt_crys=ref_crys_list)
    recon_metrics = rec_evaluator.get_metrics()

    all_metrics.update(recon_metrics)

    print(all_metrics)

    metrics_out_file = "eval_metrics.json"
    metrics_out_file = os.path.join(metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, "w") as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, "w") as f:
            json.dump(all_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()
    main(args)
