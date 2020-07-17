import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from graphae.trainer import PLGraphAE
from graphae.hyperparameter import add_arguments


#logging.getLogger("lightning").setLevel(logging.WARNING)


def main(hparams):
    if not os.path.isdir(hparams.save_dir + "/run{}/".format(hparams.id)):
        print("Creating directory")
        os.mkdir(hparams.save_dir + "/run{}/".format(hparams.id))
    print("Starting Run {}".format(hparams.id))
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.save_dir + "/run{}/".format(hparams.id),
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor='val_loss',
        mode='min',
        prefix=str(hparams.id) + '_',
        period=0
    )
    lr_logger = LearningRateLogger()
    tb_logger = TensorBoardLogger(hparams.save_dir + "/run{}/".format(hparams.id))
    model = PLGraphAE(hparams.__dict__)
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        progress_bar_refresh_rate=10 if hparams.progress_bar else 0,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=hparams.eval_freq,
        distributed_backend=None if hparams.gpus == 1 else "dp",
        gradient_clip_val=0.5,
        callbacks=[lr_logger]
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    main(args)
