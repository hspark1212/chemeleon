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
python -u run.py \ 
    with project_name="Chemeleon_v0.1.1_mineral" \
    dataset_name="mp-40-mineral" \
    data_dir="data/mp-40/mineral" \
    exp_name="chemeleon_clip_mineral" \
    group_name="mineral" \
    text_targets="mineral" \
    text_encoder="chemeleon/clip-mp-mineral" \
    batch_size=64

# evaluation
python -u chemeleon/scripts/evaluate.py \
    --model_path=hspark1212/Chemeleon_v0.1.1_mineral/model-bwy55zfv:v1 \
    --test_data=data/mp-40-mineral/test.csv \
    --save_path=Chemeleon_v0.1.1_mineral/chemeleon_clip_mineral \
    --wandb_log=True \
    --wandb_project=Chemeleon_v0.1.1_mineral \
    --wandb_group=test/prompt \
    --wandb_name=test_chemeleon_clip_mineral > eval_clip_mineral.log 2>&1
    