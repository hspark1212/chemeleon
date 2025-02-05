MODEL_NAMES = [
    "pranav-s/MaterialsBERT",  # A general-purpose material property data extraction pipeline from large polymer corpora using natural language processing
    "m3rg-iitd/matscibert",  # MatSciBERT: A materials domain language model for text mining and information extraction
    "lfoppiano/MatTPUSciBERT",  # https://huggingface.co/lfoppiano/MatTPUSciBERT (768 dim)
    "t5-3b",  # T5-3B (1024 dim)
    "meta-llama/Meta-Llama-3-8B-Instruct",  # Meta-Llama-3-8B-Instruct (4096 dim)
    "microsoft/Phi-3-mini-4k-instruct",  # Phi-3-mini-4k-instruct (3072 dim)
    "microsoft/phi-2",  # Phi-2 (2560 dim)
    "chemeleon/clip-mp-composition",
    "chemeleon/clip-mp-composition_crystalsystem",
    "chemeleon/clip-mp-prompt",
    "chemeleon/clip-mp-mineral",
    "chemeleon/clip-mp_random_split-composition",
    "chemeleon/clip-mp_random_split-prompt",
    "chemeleon/clip-mp_20-composition",
]
ARTIFACT_PATHS = {
    "chemeleon/clip-mp-composition": "hspark1212/Chemeleon_v0.1.1/model-hlfus38h:v1",
    "chemeleon/clip-mp-composition_crystalsystem": "hspark1212/Chemeleon_v0.1.1/model-b0xyc1sy:v1",
    "chemeleon/clip-mp-prompt": "hspark1212/Chemeleon_v0.1.1/model-upy53q4b:v1",
    "chemeleon/clip-mp_random_split-composition": "hspark1212/Chemeleon_v0.1.1_random_split/model-gvxwkykb:v1",
    "chemeleon/clip-mp_random_split-prompt": "hspark1212/Chemeleon_v0.1.1_random_split/model-py0nwtn6:v1",
    "chemeleon/clip-mp_20-composition": "hspark1212/Chemeleon_v0.1.1_mp-20/model-wxzdl494:v1",
    "chemeleon/clip-mp-mineral": "hspark1212/Chemeleon_v0.1.1_mineral/model-667vsj0e:v1",
}
