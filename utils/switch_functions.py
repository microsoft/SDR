from models.SDR.similarity_modeling import SimilarityModeling
import torch
import transformers
from transformers import get_linear_schedule_with_warmup
from torch import optim


def model_class_pointer(task_name, arch):
    """Get pointer to class base on flags.
    Arguments:
        task_name {str} -- the task name, node clasification etc
        arch {str} -- recobert, etc
    Raises:
        Exception: If unknown task,dataset

    Returns:
        torch.nn.Module -- The module to train on

    """

    if task_name == "document_similarity":
        if arch == "SDR":
            from models.SDR.SDR import SDR

            return SDR
    raise Exception("Unkown task")


def choose_optimizer(params, network_parameters):
    """
    Choose the optimizer from params.optimizer flag

    Args:
        params (dict): The input flags
        network_parameters (dict): from net.parameters()

    Raises:
        Exception: If not matched optimizer
    """
    if params.optimizer == "adamW":
        optimizer = transformers.AdamW(network_parameters, lr=params.lr,)
    elif params.optimizer == "sgd":
        optimizer = torch.optim.SGD(network_parameters, lr=params.lr, weight_decay=params.weight_decay, momentum=0.9,)
    else:
        raise Exception("No valid optimizer provided")
    return optimizer


def choose_scheduler(scheduler_name, optimizer, warmup_steps, params):

    if scheduler_name == "linear_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=params.max_epochs)
    elif scheduler_name == "cosine_annealing_lr":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0,)

    else:
        raise Exception("No valid optimizer provided")


from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


def choose_model_class_configuration(arch, base_model_name):
    MODEL_CLASSES = {
        "SDR_roberta": (RobertaConfig, SimilarityModeling, RobertaTokenizer),
    }
    return MODEL_CLASSES[f"{arch}_{base_model_name}"]
