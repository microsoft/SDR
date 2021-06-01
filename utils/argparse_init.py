import argparse
import sys
import os
from argparse import ArgumentDefaultsHelpFormatter


def str2intlist(v):
    if v.isdigit():
        return [int(v)]
    try:
        return [int(dig) for dig in v.split("_")]
    except Exception as e:
        raise argparse.ArgumentTypeError('Excpected int or "4_4"')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def default_arg_parser(description="", conflict_handler="resolve", parents=[], is_lowest_leaf=False):
    """
        Generate the default parser - Helper for readability
        
        Args:
            description (str, optional): name of the parser - usually project name. Defaults to ''.
            conflict_handler (str, optional): whether to raise error on conflict or resolve(take last). Defaults to 'resolve'.
            parents (list, optional): [the name of parent argument managers]. Defaults to [].
        
        Returns:
            [type]: [description]
        """
    description = (
        parents[0].description + description
        if len(parents) != 0 and parents[0] is not None and parents[0].description is not None
        else description
    )
    parser = argparse.ArgumentParser(
        description=description,
        add_help=is_lowest_leaf,
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler=conflict_handler,
        parents=parents,
    )

    return parser

def get_non_default(parsed,parser):
    non_default = {
        opt.dest: getattr(parsed, opt.dest)
        for opt in parser._option_string_actions.values()
        if hasattr(parsed, opt.dest) and opt.default != getattr(parsed, opt.dest)
    }
    return non_default

    
def init_parse_argparse_default_params(parser, dataset_name=None, arch=None):
    TASK_OPTIONS = ["document_similarity"]

    parser.add_argument(
        "--task_name", type=str, default="document_similarity", choices=TASK_OPTIONS, help="The task to solve",
    )
    task_name = parser.parse_known_args()[0].task_name

    DATASET_OPTIONS = {
        "document_similarity": ["video_games", "wines",],
    }
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DATASET_OPTIONS[task_name][0],
        choices=DATASET_OPTIONS[task_name],
        help="The dataset to evalute on",
    )
    dataset_name = dataset_name or parser.parse_known_args()[0].dataset_name

    ## General learning parameters
    parser.add_argument(
        "--train_batch_size", default={"document_similarity": 32}[task_name], type=int, help="Number of samples in batch",
    )
    parser.add_argument(
        "--max_epochs", default={"document_similarity": 50}[task_name], type=int, help="Number of epochs to train",
    )
    parser.add_argument(
        "-lr", default={"document_similarity": 2e-5}[task_name], type=float, help="Learning rate",
    )

    parser.add_argument("--optimizer", default="adamW", help="Optimizer to use")
    parser.add_argument(
        "--scheduler",
        default="linear_with_warmup",
        choices=["linear_with_warmup", "cosine_annealing_lr"],
        help="Scheduler to use",
    )
    parser.add_argument("--weight_decay", default=5e-3, help="weight decay")

    ## Input Output parameters
    parser.add_argument(
        "--default_root_dir", default=os.path.join(os.getcwd(), "output", task_name), help="The path to store this run output",
    )
    output_dir = parser.parse_known_args()[0].default_root_dir
    os.makedirs(output_dir, exist_ok=True)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    ### Model Parameters
    parser.add_argument(
        "--arch", "--architecture", default={"document_similarity": "SDR"}[task_name], help="Architecture",
    )

    architecture = arch or parser.parse_known_args()[0].arch

    parser.add_argument("--accumulate_grad_batches", default=1, type=int)

    ### Auxiliary parameters
    parser.add_argument("--gpus", default=1, type=str, help="gpu count")
    parser.add_argument("--num_data_workers", default=0, type=int, help="for parallel data load")
    parser.add_argument("--overwrite_data_cache", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument("--train_val_ratio", default=0.90, type=float, help="The split ratio of the data")
    parser.add_argument(
        "--limit_train_batches", default=10000, type=int,
    )

    parser.add_argument(
        "--train_log_every_n_steps", default=50, type=int,
    )
    parser.add_argument(
        "--val_log_every_n_steps", default=1, type=int,
    )
    parser.add_argument(
        "--test_log_every_n_steps", default=1, type=int,
    )


    parser.add_argument("--resume_from_checkpoint", default=None, type=str, help="Path to reload pretrained weights")
    parser.add_argument(
        "--metric_to_track", default=None, help="which parameter to track on saving",
    )
    parser.add_argument("--val_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--test_only", type=str2bool, nargs="?", const=True, default=False)

    return {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "architecture": architecture,
    }

