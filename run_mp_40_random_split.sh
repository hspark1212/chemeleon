# mp-40-random_split

# crystal clip (composition)
python -u run_crystal_clip.py \
    with project_name="Chemeleon_v0.1.1_random_split" \
    dataset_name="mp-40-random_split" \
    data_dir="data/mp-40/random_split" \
    exp_name="clip_composition" \
    group_name="crytal_clip" \
    text_targets="composition"

# crystal clip (prompt)
python -u run_crystal_clip.py \
    with project_name="Chemeleon_v0.1.1_random_split" \
    dataset_name="mp-40-random_split" \
    data_dir="data/mp-40/random_split" \
    exp_name="clip_prompt" \
    group_name="crytal_clip" \
    text_targets="prompt"

# diffusion (composition)
python -u run.py \
    with project_name="Chemeleon_v0.1.1_random_split" \
    dataset_name="mp-40-random_split" \
    data_dir="data/mp-40/random_split" \
    exp_name="chemeleon_clip_composition" \
    group_name="composition" \
    text_targets="composition" \
    text_encoder="chemeleon/clip-mp_random_split-composition"

# diffusion (prompt)
python -u run.py \
    with project_name="Chemeleon_v0.1.1_random_split" \
    dataset_name="mp-40-random_split" \
    data_dir="data/mp-40/random_split" \
    exp_name="chemeleon_clip_prompt" \
    group_name="prompt" \
    text_targets="prompt" \
    text_encoder="chemeleon/clip-mp_random_split-prompt"

# evaluation
python -u chemeleon/scripts/evaluate.py \
    --model_path=hspark1212/Chemeleon_v0.1.1_random_split/model-90sveydm:v1 \
    --test_data=data/mp-40/random_split/test.csv \
    --save_path=Chemeleon_v0.1.1_random_split/chemeleon_composition \
    --wandb_log=True \
    --wandb_project=Chemeleon_v0.1.1_random_split \
    --wandb_group=test/composition \
    --wandb_name=test_chemeleon_clip_comp > eval_clip_comp.log 2>&1

python -u chemeleon/scripts/evaluate.py \
    --model_path=hspark1212/Chemeleon_v0.1.1_random_split/model-nqz9nhj6:v1 \
    --test_data=data/mp-40/random_split/test.csv \
    --save_path=Chemeleon_v0.1.1_random_split/chemeleon_prompt \
    --wandb_log=True \
    --wandb_project=Chemeleon_v0.1.1_random_split \
    --wandb_group=test/prompt \
    --wandb_name=test_chemeleon_clip_prompt > eval_clip_prompt.log 2>&1