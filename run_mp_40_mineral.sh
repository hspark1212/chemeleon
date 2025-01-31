# mp-40-mineral

# crystal clip (mineral)
python -u run_crystal_clip.py \
    with project_name="Chemeleon_v0.1.1_mineral" \
    dataset_name="mp-40-mineral" \
    data_dir="data/mp-40/mineral" \
    exp_name="clip_mineral" \
    group_name="crytal_clip" \
    text_targets="mineral" \
    batch_size=64

# diffusion (mineral)
python -u run.py with project_name="Chemeleon_v0.1.1_mineral" \
    dataset_name="mp-40-mineral" \
    data_dir="data/mp-40/mineral" \
    exp_name="chemeleon_clip_mineral" \
    group_name="mineral" \
    text_targets="mineral" \
    batch_size=64

# evaluation