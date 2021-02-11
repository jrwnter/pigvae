import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from graphae.trainer import PLGraphAE
from graphae.molecular_graphs.hyperparameter import add_arguments
from graphae.molecular_graphs.data import MolecularGraphDataModule
from graphae.ddp import MyDDP
from graphae.molecular_graphs.metrics import Critic


logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_top_k=1,
        monitor="val_loss",
        save_last=True
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    critic = Critic(hparams.__dict__)
    model = PLGraphAE(hparams.__dict__, critic)
    datamodule = MolecularGraphDataModule(
        sdf_path=hparams.data_path,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        num_eval_samples=hparams.num_eval_samples,
    )
    my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        accelerator="ddp",
        plugins=[my_ddp_plugin],
        gradient_clip_val=0.1,
        callbacks=[lr_logger],
        profiler=True,
        terminate_on_nan=True,
        replace_sampler_ddp=False,
        precision=hparams.precision,
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
