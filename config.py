import envyaml

CONFIG_DIR = "config"

def load_config(config_file, preset="default", config_dir=CONFIG_DIR):
    cfg = envyaml.EnvYAML(f"{config_dir}/{config_file}")[preset]
    return cfg

