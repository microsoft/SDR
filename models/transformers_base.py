"""Stub."""
from models.doc_similarity_pl_template import DocEmbeddingTemplate
from utils import argparse_init
from utils import switch_functions


class TransformersBase(DocEmbeddingTemplate):

    """
    Author: Dvir Ginzburg.

    This is a template for future document templates using transformers.
    """

    def __init__(
        self, hparams,
    ):
        super(TransformersBase, self).__init__(hparams)
        (self.config_class, self.model_class, self.tokenizer_class,) = switch_functions.choose_model_class_configuration(
            self.hparams.arch, self.hparams.base_model_name
        )
        if self.hparams.config_name:
            self.config = self.config_class.from_pretrained(self.hparams.config_name, cache_dir=None)
        elif self.hparams.arch_or_path:
            self.config = self.config_class.from_pretrained(self.hparams.arch_or_path)
        else:
            self.config = self.config_class()
        if self.hparams.tokenizer_name:
            self.tokenizer = self.tokenizer_class.from_pretrained(self.hparams.tokenizer_name)
        elif self.hparams.arch_or_path:
            self.tokenizer = self.tokenizer_class.from_pretrained(self.hparams.arch_or_path)
        else:
            raise ValueError(
                "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name".format(self.tokenizer_class.__name__)
            )
        self.hparams.tokenizer_pad_id = self.tokenizer.pad_token_id
        self.model = self.model_class.from_pretrained(
            self.hparams.config_name, from_tf=bool(".ckpt" in self.hparams.config_name), config=self.config, hparams=self.hparams
        )

    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = DocEmbeddingTemplate.add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.add_argument(
            "--mlm",
            type=argparse_init.str2bool,
            nargs="?",
            const=True,
            default=True,
            help="Train with masked-language modeling loss instead of language modeling.",
        )

        parser.add_argument(
            "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss",
        )
        parser.add_argument(
            "--base_model_name", type=str, default="roberta", help="The underliying BERT-like model this arc.",
        )
        base_model_name = parser.parse_known_args()[0].base_model_name
        if base_model_name in ["roberta", "tnlr"]:
            default_config, default_tokenizer = "roberta-large", "roberta-large"
        elif base_model_name in ["bert", "tnlr3"]:
            default_config, default_tokenizer = "bert-large-uncased", "bert-large-uncased"
        elif base_model_name == "longformer":
            default_config, default_tokenizer = "allenai/longformer-base-4096", "allenai/longformer-base-4096"
            parser.set_defaults(batch_size=2)
        parser.add_argument(
            "--config_name",
            type=str,
            default=default_config,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=default_tokenizer,
            type=str,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
        )

        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

        parser.set_defaults(lr=2e-5, weight_decay=0)

        arch, mlm = parser.parse_known_args()[0].arch, parser.parse_known_args()[0].mlm
        if arch in ["bert", "roberta", "distilbert", "camembert", "recoberta", "recoberta_cosine"] and not mlm:
            raise ValueError(
                "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
                "flag (masked language modeling)."
            )

        return parser
