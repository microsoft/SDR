import collections
from utils.torch_utils import to_numpy
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
import torch
from utils import argparse_init
from utils import switch_functions
from utils.model_utils import extract_model_path_for_hyperparams
from subprocess import Popen
import pandas as pd


class DocEmbeddingTemplate(LightningModule):

    """
    Author: Dvir Ginzburg.

    This is a template for future document templates using pytorch lightning.
    """

    def __init__(
        self, hparams,
    ):
        super(DocEmbeddingTemplate, self).__init__()
        self.hparams = hparams
        self.hparams.hparams_dir = extract_model_path_for_hyperparams(self.hparams.default_root_dir, self)
        self.losses = {}
        self.tracks = {}
        self.hparams.mode = "val"


    def forward(self, data):
        """
        forward function for the doc similarity network
        """
        raise NotImplementedError()

    def training_step(self, batch, batch_idx, mode="train"):
        """
        Lightning calls this inside the training loop with the 
        data from the training dataloader passed in as `batch`.
        """
        self.losses = {}
        self.tracks = {}
        self.hparams.batch_idx = batch_idx
        self.hparams.mode = mode
        self.batch = batch

        batch = self(batch)

        self.tracks[f"tot_loss"] = sum(self.losses.values()).mean()

        all = {k: to_numpy(v) for k, v in {**self.tracks, **self.losses}.items()}
        getattr(self, f"{mode}_logs", None).append(all)
        self.log_step(all)

        output = collections.OrderedDict({"loss": self.tracks[f"tot_loss"]})
        return output

    def validation_step(self, batch, batch_idx, mode="val"):
        """Lightning calls this inside the validation loop with the data from the validation dataloader passed in as `batch`."""

        return self.training_step(batch, batch_idx, mode=mode)

    def log_step(self, all):
        if not (
            getattr(self.hparams, f"{self.hparams.mode}_batch_size")
            % (getattr(self.hparams, f"{self.hparams.mode}_log_every_n_steps"))
            == 0
        ):
            return
        for k, v in all.items():
            if v.shape != ():
                v = v.sum()
            self.logger.experiment.add_scalar(f"{self.hparams.mode}_{k}_step", v, global_step=self.global_step)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode="test")

    def on_validation_epoch_start(self):
        self.val_logs = []
        self.hparams.mode = "val"

    def on_train_epoch_start(self):
        self.hparams.current_epoch = self.current_epoch
        self.train_logs = []
        self.hparams.mode = "train"

    def on_test_epoch_start(self):
        self.test_logs = []
        self.hparams.mode = "test"

    def on_epoch_end_generic(self):
        if self.trainer.running_sanity_check:
            return
        logs = getattr(self, f"{self.hparams.mode}_logs", None)

        self.log_dict(logs, prefix=self.hparams.mode)

    def log_dict(self, logs, prefix):
        dict_of_lists = pd.DataFrame(logs).to_dict("list")
        for lst in dict_of_lists:
            dict_of_lists[lst] = list(filter(lambda x: not pd.isnull(x), dict_of_lists[lst]))
        for key, lst in dict_of_lists.items():
            s = 0
            for item in lst:
                s += item.sum()
            name = f"{prefix}_{key}_epoch"
            val = s / len(lst)
            self.logger.experiment.add_scalar(name, val, global_step=self.global_step)
            if self.hparams.metric_to_track == name:
                self.log(name, torch.tensor(val))

    def on_train_epoch_end(self, outputs) -> None:
        self.on_epoch_end_generic()

    def on_validation_epoch_end(self) -> None:
        if self.trainer.running_sanity_check:
            return
        if self.current_epoch % 10 == 0:
            self.logger.experiment.add_text("Profiler", self.trainer.profiler.summary(), global_step=self.global_step)


    def on_test_epoch_end(self) -> None:
        self.on_epoch_end_generic()

    def validation_epoch_end(self, outputs):
        self.on_epoch_end_generic()

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.

        At least one optimizer is required.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = switch_functions.choose_optimizer(self.hparams, optimizer_grouped_parameters)
        scheduler = switch_functions.choose_scheduler(
            self.hparams.scheduler, optimizer, warmup_steps=0, params=self.hparams
        )

        return [optimizer], [scheduler]

    def dataloader(self):
        """
        Returns the relevant dataloader (called once per training).
        
        Args:
            train_val_test (str, optional): Define which dataset to choose from. Defaults to 'train'.
        
        Returns:
            Dataset
        """
        raise NotImplementedError()

    def prepare_data(self):
        """
        Here we upload the data, called once, all the mask and train, eval split.

        Returns:
           Tuple of datasets: train,val and test dataset splits
        """
        raise NotImplementedError()

    def train_dataloader(self):
        log.info("Training data loader called.")
        return self.dataloader(mode="train")

    def val_dataloader(self):
        log.info("Validation data loader called.")
        return self.dataloader(mode="val")

    def test_dataloader(self):
        log.info("Test data loader called.")
        return self.dataloader(mode="test")

    @staticmethod
    def add_model_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        """
        Static function to add all arguments that are relevant only for this module

        Args:
            parent_parser (ArgparseManager): The general argparser
        
        Returns:
            ArgparseManager : The new argparser
        """
        parser.add_argument(
            "--test_sample_size", default=-1, type=int, help="The number of samples to eval recos on. (-1 is all)"
        )
        parser.add_argument("--top_k_size", default=-1, type=int, help="The number of top k correspondences. (-1 is all)")

        parser.add_argument("--with_same_series", type=argparse_init.str2bool, nargs="?", const=True, default=True)

        return parser

