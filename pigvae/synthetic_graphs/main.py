import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pigvae.trainer import PLGraphAE
from pigvae.synthetic_graphs.hyperparameter import add_arguments
from pigvae.synthetic_graphs.data import GraphDataModule
from pigvae.ddp import MyDDP
from pigvae.synthetic_graphs.metrics import Critic


logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_last=True,
        save_top_k=1,
        monitor="val_loss"
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    critic = Critic
    model = PLGraphAE(hparams.__dict__, critic)
    graph_kwargs = {
        "n_min": hparams.n_min,
        "n_max": hparams.n_max,
        "m_min": hparams.m_min,
        "m_max": hparams.m_max,
        "p_min": hparams.p_min,
        "p_max": hparams.p_max
    }
    datamodule = GraphDataModule(
        graph_family=hparams.graph_family,
        graph_kwargs=graph_kwargs,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        samples_per_epoch=100000000
    )
    my_ddp_plugin = MyDDP()
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=5 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=True,
        val_check_interval=hparams.eval_freq if not hparams.test else 100,
        accelerator="ddp",
        plugins=[my_ddp_plugin],
        gradient_clip_val=0.1,
        callbacks=[lr_logger, checkpoint_callback],
        terminate_on_nan=True,
        replace_sampler_ddp=False,
        precision=hparams.precision,
        max_epochs=hparams.num_epochs,
        reload_dataloaders_every_epoch=True,
        resume_from_checkpoint=hparams.resume_ckpt if hparams.resume_ckpt != "" else None
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
