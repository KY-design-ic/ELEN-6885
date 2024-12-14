!pip install -q condacolab
import condacolab
condacolab.install()


!conda create --prefix="/local_path_to_save_env/"  python=3.10
#!conda init bash
#!conda activate "/local_path_to_save_env/"
#!conda create --prefix="/content/conda_env"  python=3.10
#!conda init bash
#!conda activate "/content/conda_env"
#!pip install trl
#!pip install transformer
#!pip install torch torchvision torchaudio
#!pip install peft
!pip install numpy
!/local_path_to_save_env/bin/python -m pip install trl==0.11.3 transformers torch torchvision torchaudio
!pip install trl[peft]==0.11.3

!pip uninstall numpy -y
!pip install numpy==1.22.4

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

tqdm.pandas()
set_seed(42)