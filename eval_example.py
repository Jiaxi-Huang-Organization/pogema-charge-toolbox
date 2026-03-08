from pogema_toolbox.evaluator import evaluation
from pogema import BatchAStarAgent

from pathlib import Path

import yaml

from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.registry import ToolboxRegistry

PROJECT_NAME = 'pogema-charge-toolbox'
BASE_PATH = Path('config_examples')


def main():
    ToolboxRegistry.setup_logger(level='INFO')
    env_cfg_name = 'Pogema-v0'
    ToolboxRegistry.register_env(env_cfg_name, create_env_base, Environment)
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)
    #ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)
    folder_names = [
        'logo',
        #'pathfinding'
    ]

    for folder in folder_names:
        #maps_path = BASE_PATH / folder / "maps.yaml"
        #with open(maps_path, 'r') as f:
        #    maps = yaml.safe_load(f)
        #ToolboxRegistry.register_maps(maps)
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        ToolboxRegistry.info(f'Starting: {folder}')

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        evaluation(evaluation_config, eval_dir=eval_dir)



if __name__ == '__main__':
    main()
