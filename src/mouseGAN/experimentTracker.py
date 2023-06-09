
import wandb
import dataclasses

PROJECT_NAME = "mouseGAN"

def initialize_wandb(config, tempProjectName=None):
    """
    need to convert config to dict to pass to wandb.init
    """
    config = dataclasses.asdict(config)
    preppedConfig = {}
    for k, v in config.items():
        if dataclasses.is_dataclass(v):
            preppedConfig[k] = dataclasses.asdict(v)
        else:
            preppedConfig[k] = v
    return wandb.init(project=PROJECT_NAME if tempProjectName is None else tempProjectName, job_type="train", config=config)
