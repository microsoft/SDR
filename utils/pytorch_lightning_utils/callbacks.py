from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer


class RunValidationOnStart(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer: Trainer, pl_module):
        return trainer.run_evaluation()
