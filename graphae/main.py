import os
import logging
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from graphae.trainer import PLGraphAE
from graphae.hyperparameter import add_arguments
from graphae.data import MolecularGraphDataModule
from graphae.ddp import MyDDP


logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    #torch.set_num_threads(8)
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_top_k=1,
        monitor="val_hard_loss",
        save_last=True
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    model = PLGraphAE(hparams.__dict__)
    datamodule = MolecularGraphDataModule(
        data_path=hparams.data_path,
        batch_size=hparams.batch_size,
        max_num_nodes=hparams.max_num_nodes,
        num_eval_samples=hparams.num_eval_samples,
        num_samples_per_epoch=hparams.num_samples_per_epoch,
        num_samples_per_epoch_inc=hparams.num_samples_per_epoch_inc,
        num_workers=hparams.num_workers,
        debug=hparams.test
    )
    my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=10 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=hparams.eval_freq if not hparams.test else 1.0,
        accelerator="ddp",
        plugins=[my_ddp_plugin],
        #gradient_clip_val=0.1,
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
