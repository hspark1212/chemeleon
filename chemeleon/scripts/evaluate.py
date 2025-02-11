from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from fire import Fire
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mace.calculators import mace_mp

import pytorch_lightning as pl
import wandb
from chemeleon.modules.chemeleon import Chemeleon


def test_evaluate(
    model_path: str | Path,
    test_data: str | Path = "data/mp-40/test.csv",
    n_samples: int = 20,
    cond_scale: float = 2.0,
    mace_dtype: str = "float32",
    mace_device: str = "cuda",
    save_path: str | Path = "results",
    wandb_log: bool = False,
    wandb_project: str = "Chemeleon_test",
    wandb_group: str = "test",
    wandb_name: str = "test",
):
    """Evaluate the model on the test set with n sampling iterations.

    0. valid_samples: how many valid samples are generated.
    1. Unique: how many unique structures are generated out of n samples.
    2. Structure Matching: the generated structures include the ground truth.
    3. Meta Stable: how many the generated structures have lower energy than the ground truth.
    4. Composition Matching: the generated structures have the same composition as the ground truth.
    5. Crystal System Matching: the generated structures have the same crystal system as the ground truth.
    6. Lattice System Matching: the generated structures have the same lattice system as the ground truth.
    """
    pl.seed_everything(42)

    # set model path
    if model_path.startswith("hspark1212"):
        print(f"Downloading model from {model_path} in wandb")
        api = wandb.Api()
        artifact = api.artifact(model_path)
        # download the artifact
        model_id = model_path.split("/")[-1]
        artifact_path = Path(".cache/") / model_id
        artifact.download(artifact_path)
        model_path = artifact_path / "model.ckpt"
    else:
        model_path = Path(model_path)

    # load model
    print(f"Model loaded from {model_path}")
    model = Chemeleon.load_from_checkpoint(model_path)  # pylint: disable=E1120
    model.eval()
    text_targets = model.hparams.text_targets
    if isinstance(text_targets, str):
        text_targets = [text_targets]
    print(f"Text targets: {text_targets}")

    # read test data
    path_test_data = Path(test_data)
    if not path_test_data.exists():
        raise FileNotFoundError(f"{path_test_data} does not exist.")
    df_test = pd.read_csv(path_test_data)
    print(f"{len(df_test)} test data loaded from {path_test_data}.")

    # create save path
    path_save = Path(save_path)
    path_save.mkdir(parents=True, exist_ok=True)

    # skip if already evaluated materials
    path_results = path_save / "results.csv"
    if path_results.exists():
        df_results = pd.read_csv(path_results)
        evaluated_materials = df_results["material_id"].values
        df_test = df_test[~df_test["material_id"].isin(evaluated_materials)]
        print(
            f"Found {len(evaluated_materials)} evaluated materials. {len(df_test)} will be evaluated."
        )
    else:
        df_results = pd.DataFrame()

    # set mace calculator
    mace_calc = mace_mp(default_dtype=mace_dtype, device=mace_device)

    # start evaluation
    for i, row in tqdm(df_test.iterrows()):
        collections = defaultdict(list)
        print(f"Evaluate {i} structure ({row['material_id']})...")
        # get test structure
        test_st = Structure.from_str(row["cif"], fmt="cif")

        # set text
        properties = row[text_targets].values
        if len(text_targets) == 1:
            text = str(properties[0])
        else:
            text = ", ".join(
                [f"{text_targets[i]}: {properties[i]}" for i in range(len(properties))]
            )

        try:
            # sample
            natoms = len(test_st)
            gen_atoms_list = model.sample(
                text_input=text,
                n_atoms=natoms,
                n_samples=n_samples,
                cond_scale=cond_scale,
            )
            gen_st_list = [
                AseAtomsAdaptor.get_structure(atoms) for atoms in gen_atoms_list
            ]
            valid_gen_st_list = test_valid(gen_st_list)
            if len(valid_gen_st_list) == 0:
                print("No valid samples generated.")
                continue

            # 1. unique
            num_unique = test_unique(valid_gen_st_list)

            # 2. structure matching
            num_match = test_structure_matching(valid_gen_st_list, test_st)

            # 3. meta stable
            meta_stable = test_meta_stable(valid_gen_st_list, test_st, mace_calc)

            # 4. composition matching
            num_comp_match = test_composition_matching(valid_gen_st_list, test_st)

            # 5. crystal system matching
            num_crystal_match = test_crystal_system_matching(valid_gen_st_list, test_st)

            # 6. lattice system matching
            num_lattice_match = test_lattice_system_matching(valid_gen_st_list, test_st)

            # collect results
            collections["material_id"].append(row["material_id"])
            collections["natoms"].append(natoms)
            collections["valid_samples"].append(
                len(valid_gen_st_list) / len(gen_st_list)
            )
            collections["unique"].append(num_unique / len(valid_gen_st_list))
            collections["structure_matching"].append(num_match > 0)
            collections["structure_matching_ratio"].append(
                num_match / len(valid_gen_st_list)
            )
            collections["meta_stable"].append(meta_stable)
            collections["composition_matching"].append(
                num_comp_match / len(valid_gen_st_list)
            )
            collections["crystal_system_matching"].append(
                num_crystal_match / len(valid_gen_st_list)
            )
            collections["lattice_system_matching"].append(
                num_lattice_match / len(valid_gen_st_list)
            )
            collections["ref_structure"].append(test_st.to(fmt="cif"))
            for n, st in enumerate(gen_st_list):
                collections[f"gen_structure_{n}"].append(st.to(fmt="cif"))

            # append results
            df_row = pd.DataFrame(collections)
            df_results = pd.concat([df_results, df_row], ignore_index=True)
            if len(df_results) % 10 == 0:
                df_results.to_csv(path_results, index=False)

        except Exception as e:  # pylint: disable=W0703
            print(f"Error: {e}")

    # mean results
    log_dict = {}
    for k in df_results.columns:
        if k.startswith("gen_structure") or k == "ref_structure" or k == "material_id":
            continue
        mean_value = np.nanmean(df_results[k])
        df_results[f"mean_{k}"] = mean_value
        log_dict[f"mean_{k}"] = mean_value
    df_results.to_csv(path_results, index=False)

    # log
    if wandb_log:
        wandb.init(project=wandb_project, group=wandb_group, name=wandb_name)
        # log for only mean values
        wandb.log(log_dict)
        wandb.save(str(path_save / "results.csv"))
        wandb.finish()


def test_valid(gen_st_list: list[Structure]):
    valid_gen_st_list = []
    for st in gen_st_list:
        # check if the lattice length < 60A
        if max(st.lattice.abc) > 60:
            continue
        # check if the lowest distance between atoms > 0.5A
        dist_mat = st.distance_matrix
        lowest_dist = np.min(dist_mat[dist_mat > 0])
        if lowest_dist < 0.5:
            continue
        valid_gen_st_list.append(st)
    return valid_gen_st_list


def test_unique(st_list: list[Structure]):
    sm = StructureMatcher()
    output_sm = sm.group_structures(st_list)
    return len(output_sm)


def test_structure_matching(st_list: list[Structure], ref_st: Structure):
    sm = StructureMatcher()
    num_match = 0
    for st in st_list:
        if sm.fit(ref_st, st):
            num_match += 1
    return num_match


def test_meta_stable(st_list: list[Structure], ref_st: Structure, mace_calc):
    ref_energy = mace_calc.get_potential_energy(ref_st.to_ase_atoms())
    num_meta_stable = 0
    num_same_comp = 0
    for st in st_list:
        if st.composition != ref_st.composition:
            continue
        num_same_comp += 1
        # calculate energy difference per atom
        gen_energy = mace_calc.get_potential_energy(st.to_ase_atoms())
        e_diff_per_atoms = (gen_energy - ref_energy) / len(st)
        if e_diff_per_atoms < 0.1:
            num_meta_stable += 1
    return num_meta_stable / num_same_comp if num_same_comp > 0 else np.NaN


def test_composition_matching(st_list: list[Structure], ref_st: Structure):
    num_match = 0
    for st in st_list:
        if ref_st.composition == st.composition:
            num_match += 1
    return num_match


def test_crystal_system_matching(
    st_list: list[Structure], ref_st: Structure, symprec=0.1, angle_tolerance=10
):
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_crystal_system = ref_sga.get_crystal_system()
    for st in st_list:
        try:
            sga = SpacegroupAnalyzer(
                st, symprec=symprec, angle_tolerance=angle_tolerance
            )
            crystal_system = sga.get_crystal_system()
            if crystal_system == ref_crystal_system:
                num_match += 1
        except Exception as e:  # pylint: disable=W0718
            print(e)
    return num_match


def test_lattice_system_matching(
    st_list: list[Structure], ref_st: Structure, symprec=0.1, angle_tolerance=10
):
    num_match = 0
    ref_sga = SpacegroupAnalyzer(
        ref_st, symprec=symprec, angle_tolerance=angle_tolerance
    )
    ref_lattice_system = ref_sga.get_lattice_type()
    for st in st_list:
        test_st = Structure(
            lattice=st.lattice,
            species=["H"],
            coords=[[0.5, 0.5, 0.5]],
        )
        sga = SpacegroupAnalyzer(
            test_st, symprec=symprec, angle_tolerance=angle_tolerance
        )
        lattice_system = sga.get_lattice_type()
        if lattice_system == ref_lattice_system:
            num_match += 1
    return num_match


if __name__ == "__main__":
    Fire(test_evaluate)
