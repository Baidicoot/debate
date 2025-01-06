import yaml

def load_prompts(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)