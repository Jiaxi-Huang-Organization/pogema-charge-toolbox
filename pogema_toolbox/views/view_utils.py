from typing import Union, List

import pandas as pd
import yaml
from pydantic import BaseModel
from collections import defaultdict
from tabulate import tabulate

import json

from pogema_toolbox.registry import ToolboxRegistry


class View(BaseModel):
    drop_keys: list = ['seed']
    round_digits: int = 2
    rename_fields: dict = {"algorithm": "Algorithm",
                           "num_agents": "Number of Agents",
                           "num_charges": "Number of Charges",
                           "avg_throughput": "Average Throughput",
                           "avg_relative_battery": "Average Battery Precentage",
                           "avg_throughput_with_active": "Average Throughput with Active Agents",
                           "avg_goal_battery_relative": "Average Goal Battery Percentage",
                           "avg_charging_per_agent": "Average Charging Per Agent",
                           'avg_charging_rate': "Average Charged Percentage",
                           "runtime": "Runtime (seconds)",
                           "valid_episode_relative": "Average Active Episode Percentage",
                           "max_episode_steps": "Max Episode Length"
                           }
    rename_algorithms: dict = {}
    sort_by: Union[str, List[str]] = None


def drop_na(df):
    # Drop columns with any NaN values
    cols_before_drop_na = set(df.columns)
    df = df.dropna(axis=1, how='any')
    cols_after_drop_na = set(df.columns)

    # Log dropped columns
    dropped_cols = cols_before_drop_na - cols_after_drop_na
    for col in dropped_cols:
        ToolboxRegistry.warning(f"Column '{col}' dropped due to missing values")

    return df


def eval_logs_to_pandas(eval_configs):
    data = {}
    for idx, config in enumerate(eval_configs):
        data[idx] = {**config['env_grid_search'], 'algorithm': config['algorithm']}

        # Adding metrics separately to skip possible lists of metrics (e.g. every step throughput)
        for key, value in config['metrics'].items():
            if isinstance(value, list):
                continue
            data[idx][key] = value
    return pd.DataFrame.from_dict(data, orient='index')


def load_from_folder(folder_path):
    eval_config_path = folder_path / (folder_path.name + '.yaml')
    # check if the file with name current_dir_name.name + '.yaml' exists
    assert eval_config_path.exists(), f'Config file {eval_config_path} does not exist'

    with open(eval_config_path, 'r') as f:
        evaluation_config = yaml.safe_load(f)

    results = []
    # Load results from *.json files
    for file in folder_path.glob('*.json'):
        with open(file, 'r') as f:
            results += json.load(f)
    return results, evaluation_config


def check_seeds(results):
    seed_data = defaultdict(lambda: defaultdict(list))
    algorithms = set()
    problem_count = 0  # Initialize problem counter

    # Populate seed_data and algorithms set
    for res in results:
        if 'env_grid_search' not in res:
            ToolboxRegistry.error("env_grid_search data missing")
            return

        env_data = res.get('env_grid_search', {})

        if 'map_name' not in env_data:
            ToolboxRegistry.error("No map_name in env_grid_search data")
            return

        if 'num_agents' not in env_data:
            ToolboxRegistry.error("No num_agents in env_grid_search data")
            return
        if 'max_episode_steps' not in env_data:
            ToolboxRegistry.error("No max_episode_steps in env_grid_search data")
            return
        if 'agent_per_charge' not in env_data:
            ToolboxRegistry.error("No agent_per_charge in env_grid_search data")
            return  
        if 'seed' not in env_data:
            ToolboxRegistry.error("No seed in env_grid_search data")
            return

        map_name = res['env_grid_search']['map_name']
        num_agents = res['env_grid_search']['num_agents']
        max_episode_steps = res['env_grid_search']['max_episode_steps']
        agent_per_charge = res['env_grid_search']['agent_per_charge']
        seed = res['env_grid_search']['seed']
        algo = res['algorithm']
        algorithms.add(algo)
        seed_data[(map_name, num_agents, seed)][algo].append(res)

    # Prepare data for tabulate
    tabulate_data = []
    headers = ['Map Name', 'Num Agents', 'Seed'] + sorted(algorithms)

    for (map_name, num_agents, seed), algos in seed_data.items():
        row_issues = False
        row = [map_name, num_agents, seed]
        for algo in sorted(algorithms):
            if algo in algos:
                if len(algos[algo]) > 1:  # Duplicated seeds for this algo
                    row.append(f"x{len(algos[algo])}")
                    problem_count += 1  # Increment problem counter
                    row_issues = True
                else:
                    row.append("ok")  # Seed present and not duplicated
            else:
                row.append("missing")  # Seed missing for this algo
                problem_count += 1  # Increment problem counter
                row_issues = True

        if row_issues:  # Add row only if there are issues
            tabulate_data.append(row)

    # If problems detected, show error with tabulate
    if problem_count > 0:
        table = tabulate(tabulate_data, headers=headers, tablefmt='psql')
        ToolboxRegistry.error(f"Detected {problem_count} problems with seeds:\n" + table)
    else:
        ToolboxRegistry.success(f"Passed seeds consistency check")
