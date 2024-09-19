# assignment/config.py

from dataclasses import dataclass


@dataclass
class Config:
    """
    We store our runtime arguments in our config for easy state management
    """
    wikipedia_dataset_path: str = "wikipedia-summary-128k.tsv" # this path is relative to your python working directory
    sampled_output_path: str = "processed-data.tsv"
    target_sample_size: int = 65536 # 2**16 - you should go more or less depending on your compute power
    random_seed: int = 42 # We want to control our random state during development so errors are reporducable
    num_lda_topics: int = 10 # The number of topics for LDA to generate
    num_lda_passes: int = 10 # Number of LDA passes
