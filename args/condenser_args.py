from dataclasses import dataclass , field
@dataclass
class beitArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    batch_size: Optional[int] = field(
        default=64
    )
    epochs: Optional[int] = field(
        default=300
    )
    save_ckpt_freq: Optional[int] = field(
        default=20
    )
    tokenizer_weight :Optional[str] = field(
        default=None
    )
    tokenizer_model:Optional[str] = field(
        default="vqkd_encoder_base_decoder_3x768x12_clip"
    )
    model: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    n_head_layers: int = field(default=2)
    skip_from: int = field(default=2)
    late_mlm: bool = field(default=False)