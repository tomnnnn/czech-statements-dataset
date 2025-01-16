import envyaml
from dotenv import load_dotenv

CONFIG_DIR = "config"

def load_config(config_file, preset="default", config_dir=CONFIG_DIR):
    load_dotenv()
    cfg = envyaml.EnvYAML(f"{config_dir}/{config_file}")[preset]
    return cfg
