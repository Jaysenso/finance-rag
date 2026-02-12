import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Usage
config = load_config()
llm_config = config["llm"]