import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import argparse
from .logging_config import setup_logging
import pprint

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
logger = logging.getLogger(__name__)

@dataclass
class Config:
    out_folder: str
    model_name: str
    index: int = 0
    max: int = 1
    with_explanation: bool = False
    prompt_config: str = ""
    example_count: int = 0
    batch_size: int = 1
    test_portion: Optional[float] = None
    allowed_labels: List[str] = field(default_factory=lambda: ["pravda", "nepravda"])
    model_api: str = "transformers"
    dataset_path: str = ""
    log_path: str = ""
    evidence_source: str = "demagog"
    model_file: Optional[str] = None
    stratify: bool = False
    relevancy_threshold: int = 1
    relevant_paragraph: bool = False
    min_evidence_count: int = 1

def load_yaml_config(path: str) -> dict:
    """Loads the configuration from a YAML file."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file {path} not found. Using defaults.")
        return {}

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-o", "--out-folder", type=str, help="Path to output folder.")
    parser.add_argument("-i", "--index", type=int, default=0, help="Index for parallelization.")
    parser.add_argument("-m", "--max", type=int, default=1, help="Maximum number of parallel processes.")
    parser.add_argument("-e", "--with-explanation", action="store_true", help="Require explanation in the model output")
    parser.add_argument("-p", "--prompt-config", type=str, help="Prompt config file path")
    parser.add_argument("-b", "--batch-size", type=int, help="Inference batch size")
    parser.add_argument("-r", "--relevancy-threshold", type=int, help="Minimum relevancy score for evidence inclusion")
    parser.add_argument("-P", "--relevant-paragraph", action="store_true", help="Include only relevant paragraph from evidence")
    parser.add_argument("-t", "--test-portion", type=float, help="Portion of dataset to sample for testing")
    parser.add_argument("-d", "--dataset-path", type=str, help="Path to dataset file")
    parser.add_argument("-a", "--allowed-labels", nargs="+", help="Labels to include in evaluation")
    parser.add_argument("-A", "--model-api", help="LLM API to use")
    parser.add_argument("-c", "--example-count", type=int, help="Number of examples per label")
    parser.add_argument("-l", "--log-path", type=str, help="Path to log file")
    parser.add_argument("-E", "--evidence-source", type=str, help="Source of evidence data")
    parser.add_argument("--model-file", type=str, help="Optional path to model file")
    parser.add_argument("model_name", type=str, nargs="?", help="Name of the model to evaluate")
    parser.add_argument("--stratify", action="store_true", help="Stratify test set by labels")
    parser.add_argument("--min-evidence-count", type=int, help="Minimum number of evidence items per example")

    args = parser.parse_args()

    if args.index and args.max is None:
        parser.error("--index requires --max.")

    return args

def merge_config(cli_args: argparse.Namespace, yaml_config: dict) -> Config:
    """Merges YAML config with CLI arguments, giving priority to CLI values."""
    config_dict = {key: value for key, value in vars(cli_args).items() if value is not None}
    merged_config = {**yaml_config, **config_dict}  # CLI args override YAML
    return Config(**merged_config)

def load_config() -> Config:
    """Loads configuration from YAML and merges it with command-line arguments."""
    yaml_config = load_yaml_config(CONFIG_PATH)
    cli_args = parse_args()
    config = merge_config(cli_args, yaml_config)

    setup_logging(config.log_path)
    logger.info(f"Loaded configuration:\n{pprint.pformat(config, indent=4)}")
    return config
