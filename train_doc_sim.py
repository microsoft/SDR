"""Top level file, parse flags and call trining loop."""
import os
from utils.pytorch_lightning_utils.pytorch_lightning_utils import load_params_from_checkpoint
import torch
from pytorch_lightning.profiler.profilers import SimpleProfiler
from utils.pytorch_lightning_utils.callbacks import RunValidationOnStart
from utils import switch_functions
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.argparse_init import default_arg_parser, init_parse_argparse_default_params
import logging

logging.basicConfig(level=logging.INFO)
from pytorch_lightning.loggers import TensorBoardLogger

def main():
    """Initialize all the parsers, before training init."""
    parser = default_arg_parser()
    parser = Trainer.add_argparse_args(parser)  # Bug in PL
    parser = default_arg_parser(description="docBert", parents=[parser])

    eager_flags = init_parse_argparse_default_params(parser)
    model_class_pointer = switch_functions.model_class_pointer(eager_flags["task_name"], eager_flags["architecture"])
    parser = model_class_pointer.add_model_specific_args(parser, eager_flags["task_name"], eager_flags["dataset_name"])

    hyperparams = parser.parse_args()
    main_train(model_class_pointer, hyperparams,parser)


def main_train(model_class_pointer, hparams,parser):
    """Initialize the model, call training loop."""
    pytorch_lightning.utilities.seed.seed_everything(seed=hparams.seed)

    if(hparams.resume_from_checkpoint not in [None,'']):
        hparams = load_params_from_checkpoint(hparams, parser)

    model = model_class_pointer(hparams)


    logger = TensorBoardLogger(save_dir=model.hparams.hparams_dir,name='',default_hp_metric=False)
    logger.log_hyperparams(model.hparams, metrics={model.hparams.metric_to_track: 0})
    print(f"\nLog directory:\n{model.hparams.hparams_dir}\n")

    trainer = pytorch_lightning.Trainer(
        num_sanity_val_steps=2,
        gradient_clip_val=hparams.max_grad_norm,
        callbacks=[RunValidationOnStart()],
        checkpoint_callback=ModelCheckpoint(
            save_top_k=3,
            save_last=True,
            mode="min" if "acc" not in hparams.metric_to_track else "max",
            monitor=hparams.metric_to_track,
            filepath=os.path.join(model.hparams.hparams_dir, "{epoch}"),
            verbose=True,
        ),
        logger=logger,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        distributed_backend="dp",
        limit_val_batches=hparams.limit_val_batches,
        limit_train_batches=hparams.limit_train_batches,
        limit_test_batches=hparams.limit_test_batches,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        profiler=SimpleProfiler(),
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        reload_dataloaders_every_epoch=True,
        # load
        resume_from_checkpoint=hparams.resume_from_checkpoint,
    )
    if(not hparams.test_only):
        trainer.fit(model)
    else:
        if(hparams.resume_from_checkpoint is not None):
            model = model.load_from_checkpoint(hparams.resume_from_checkpoint,hparams=hparams, map_location=torch.device(f"cpu"))
    trainer.test(model)


if __name__ == "__main__":
    main()

