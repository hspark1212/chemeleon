import os
import copy
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from chemeleon.config import ex
from chemeleon.datamodule import DataModule
from chemeleon.text_encoder.crystal_clip import CrystalClip


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    pl.seed_everything(_config["seed"])
    project_name = _config["project_name"]
    current_time = datetime.now().strftime("%Y-%m-%d")
    exp_name = (
        f"test_{_config['exp_name']}_{current_time}"
        if _config["test_only"]
        else f"{_config['exp_name']}_{current_time}"
    )
    log_dir = Path(_config["log_dir"], _config["dataset_name"])
    os.environ["WANDB_DIR"] = str(log_dir)
    offline = _config["offline"]

    # set datamodule
    dm = DataModule(_config)

    # set model
    module = CrystalClip(_config)
    print(module)

    # set checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        filename="best-{epoch}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=_config["early_stopping"],
        verbose=True,
        mode="min",
    )
    callbacks = [
        checkpoint_callback,
        lr_callback,
        early_stop_callback,
    ]

    # set logger
    logger = WandbLogger(
        project=project_name,
        name=exp_name,
        offline=offline,
        save_dir=log_dir,
        log_model=True if not offline else False,
        group=_config["group_name"],
    )

    # set trainer
    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        devices=_config["devices"],
        accelerator=_config["accelerator"],
        max_epochs=_config["max_epochs"],
        strategy="ddp_find_unused_parameters_true",
        deterministic=_config["deterministic"],
        gradient_clip_val=_config["gradient_clip_val"],
        limit_test_batches=_config["limit_test_batches"],
        accumulate_grad_batches=_config["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=logger,
    )

    # run
    if _config["test_only"]:
        trainer.test(module, datamodule=dm, ckpt_path=["resume_from"])
    else:
        trainer.fit(module, datamodule=dm, ckpt_path=_config["resume_from"])
        trainer.test(datamodule=dm, ckpt_path="best")
