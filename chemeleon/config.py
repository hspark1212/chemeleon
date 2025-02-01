# pylint: disable-all
from sacred import Experiment

ex = Experiment("chemeleon")


@ex.config
def config():
    # base
    project_name = "Chemeleon_v0.2.1"  # project_name for wandb
    exp_name = "chemeleon"
    group_name = "mp-40"
    seed = 0
    test_only = False
    offline = False  # offline mode for wandb
    sweep = False  # sweep mode for wandb

    # dataset
    dataset_name = "mp-40"
    data_dir = "data/mp-40"

    # dataloader
    batch_size = 128
    num_workers = 0
    pin_memory = True

    # decoder
    hidden_dim = 512
    time_dim = 128
    text_dim = 512
    max_atoms = 103 + 1  # 103 atoms + 1 for dummy atom
    num_layers = 6
    act_fn = "silu"
    dis_emb = "sin"
    num_freqs = 128
    edge_style = "fc"
    max_neighbors = 20
    cutoff = 6.0
    ln = True
    ip = True
    smooth = False
    pred_atom_types = True

    # chemeleon
    text_guide = True
    text_targets = [
        "composition"
    ]  # "composition", "crystal_system", "space_group", "dimensionality", "general_text"
    trainable_text_encoder = False
    text_encoder = "lfoppiano/MatTPUSciBERT"  # "pranav-s/MaterialsBERT", "m3rg-iitd/matscibert", "lfoppiano/MatTPUSciBERT"
    text_embed_dim = 768  # embedding dimension of text encoder
    max_text_len = 256  # max length of text guide
    cond_drop_prob = 0.2  # conditional drop probability
    beta_schedule = "cosine"  # "cosine", "linear", "quadratic", "sigmoid"
    timesteps = 1000
    max_num_atoms = 50
    cost_atom_types = 1.0  # loss weight for atom types
    cost_lattice = 1.0  # loss weight for lattice
    cost_coords = 1.0  # loss weight for coords
    d3pm_hybrid_coeff = 1.0  # hybrid coefficient for D3PM

    # crystal-clip (contrastive learning)
    clip_dim = 768
    label_smoothing = 0.1
    graph_pooling = "mean"
    graph_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    accumulate_grad_batches = 1

    # optimizer
    optimizer = "adam"  # "adam", "sgd", "adamw"
    lr = 1e-3  # learning rate
    weight_decay = 0
    scheduler = "reduce_on_plateau"  # "constant", "cosine", "reduce_on_plateau", "constant_with_warmup"
    patience = 200  # patience for reduce_on_plateau scheduler
    early_stopping = 300  # patience for early stopping

    # training
    num_nodes = 1  # number of nodes for distributed training
    devices = 1  # number of GPUs to use
    accelerator = "gpu"  # "cpu", "gpu"
    max_epochs = 1000
    deterministic = True  # set True for reproducibility
    log_dir = "./logs"
    load_path = None  # to load pretrained model
    resume_from = None  # resume from checkpoint
    gradient_clip_val = 0.5  # 0.0 for no clipping
    limit_test_batches = 1.0  # for faster testing

    # test evaluation
    cond_scale = 2.0  # scale for conditional sampling
    meta_stable_test = True
    dynamic_stable_test = True
    optimization_test = False  # it takes too long to run, so default is False
    wandb_id = None


################
# Crystal Clip #
################
@ex.named_config
def clip_composition():
    exp_name = "clip_composition"
    group_name = "crystal_clip"

    text_targets = ["composition"]


@ex.named_config
def clip_crystal_system():
    exp_name = "clip_crystal_system"
    group_name = "crystal_clip"

    text_targets = ["crystal_system"]


@ex.named_config
def clip_composition_crystal_system():
    exp_name = "clip_composition_crystal_system"
    group_name = "crystal_clip"

    text_targets = ["composition", "crystal_system"]


@ex.named_config
def clip_composition_stability():
    exp_name = "clip_composition_stability"
    group_name = "crystal_clip"

    text_targets = ["composition", "stability"]


@ex.named_config
def clip_composition_metallic():
    exp_name = "clip_composition_metallic"
    group_name = "crystal_clip"

    text_targets = ["composition", "metallic"]


@ex.named_config
def clip_prompt():
    exp_name = "clip_prompt"
    group_name = "crystal_clip"

    text_targets = ["prompt"]


#########
# mp-40 #
#########


@ex.named_config
def unguided():
    exp_name = "unguided"
    group_name = "unguided"

    text_guide = False
    text_targets = []


#######################
##### composition #####
########################


@ex.named_config
def chemeleon_bert_composition():
    exp_name = "chemeleon_bert_composition"
    group_name = "composition"

    text_targets = ["composition"]


@ex.named_config
def chemeleon_clip_composition():
    exp_name = "chemeleon_clip_composition"
    group_name = "composition"

    text_targets = ["composition"]
    text_encoder = "chemeleon/clip-mp-composition"


##########################
##### crystal_system #####
##########################


@ex.named_config
def chemeleon_bert_crystal_system():
    exp_name = "chemeleon_bert_crystal_system"
    group_name = "crystal_system"

    text_targets = ["crystal_system"]


@ex.named_config
def chemeleon_clip_crystal_system():
    exp_name = "chemeleon_clip_crystal_system"
    group_name = "crystal_system"

    text_targets = ["crystal_system"]
    text_encoder = "chemeleon/clip-mp-crystalsystem"


########################################
##### composition + crystal_system #####
########################################


@ex.named_config
def chemeleon_bert_composition_crystal_system():
    exp_name = "chemeleon_bert_composition_crystal_system"
    group_name = "composition_crystal_system"

    text_targets = ["composition", "crystal_system"]


@ex.named_config
def chemeleon_clip_composition_crystal_system():
    exp_name = "chemeleon_clip_composition_crystal_system"
    group_name = "composition_crystal_system"

    text_targets = ["composition", "crystal_system"]
    text_encoder = "chemeleon/clip-mp-composition_crystalsystem"


##################
##### prompt #####
##################


@ex.named_config
def chemeleon_clip_prompt():
    exp_name = "chemeleon_clip_prompt"
    group_name = "prompt"

    text_targets = ["prompt"]
    text_encoder = "chemeleon/clip-mp-prompt"


###################################
##### composition + stability #####
###################################


@ex.named_config
def chemeleon_clip_composition_stability():
    exp_name = "chemeleon_clip_composition_stability"
    group_name = "composition_stability"

    text_targets = ["composition", "stability"]
    text_encoder = "chemeleon/clip-mp-composition_stablity"


###################################
##### composition + metallic #####
###################################


@ex.named_config
def chemeleon_clip_composition_metallic():
    exp_name = "chemeleon_clip_composition_metallic"
    group_name = "composition_metallic"

    text_targets = ["composition", "metallic"]
    text_encoder = "chemeleon/clip-mp-composition_metallic"
