"""gLM2 model configuration"""

from typing import Optional
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class gLM2Config(PretrainedConfig):
    model_type = "gLM2"

    def __init__(
        self,
        dim: int = 640,
        depth: int = 30,
        heads: int = 10,
        vocab_size: int = 4160,
        swiglu_multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.vocab_size = vocab_size
        self.swiglu_multiple_of = swiglu_multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps

        self.auto_map = {
            "AutoConfig": "configuration_glm2.gLM2Config",
            "AutoModel": "modeling_glm2.gLM2Model",
            "AutoModelForMaskedLM": "modeling_glm2.gLM2ForMaskedLM"
        }
