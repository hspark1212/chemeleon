# mp-20

# crystal clip (composition)
python -u run_crystal_clip.py \
    with project_name="Chemeleon_v0.1.1_mp-20" \
    dataset_name="mp-20" \
    data_dir="data/mp-20" \
    exp_name="clip_composition" \
    group_name="crytal_clip" \
    text_targets="composition"

# diffusion (composition)
python -u run.py \
    with project_name="Chemeleon_v0.1.1_mp-20" \
    dataset_name="mp-20" \
    data_dir="data/mp-20" \
    exp_name="chemeleon_clip_composition" \
    group_name="composition" \
    text_targets="composition" \
    text_encoder="chemeleon/clip-mp_20-composition"

# evaluation
python -u chemeleon/scripts/evaluate.py \
    --model_path=hspark1212/Chemeleon_v0.1.1_mp-20/model-j3wbw9k9:v1 \
    --test_data=data/mp-20/test.csv \
    --save_path=Chemeleon_v0.1.1_mp-20/chemeleon_composition \
    --wandb_log=True \
    --wandb_project=Chemeleon_v0.1.1_mp-20 \
    --wandb_group=test/composition \
    --wandb_name=test_chemeleon_clip_comp > eval_clip_comp_mp-20.log 2>&1