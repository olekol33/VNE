from functools import lru_cache
from include import config
from pathlib import Path
import os
import glob
import shutil

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FileHandler:
    @staticmethod
    def create_exp_rundir(run_number, exp_name):
        runpath = FileHandler.get_run_dir(run_number, exp_name)
        if not os.path.exists(runpath):
            FileHandler.create_run_dir(run_number, exp_name)

    @staticmethod
    def create_run_dir(run_number: int = None, exp_name: str = None, clear=False):
        scenario_path = FileHandler.get_run_dir(run_number, exp_name)
        if clear:
            FileHandler.clear_create_path(scenario_path)
        if run_number is None and exp_name is None:
            shutil.copy(Path(os.path.dirname(__file__)) / 'include' / 'config.py', scenario_path)

    @staticmethod
    def get_run_dir(run=None, exp=None, topology=config.topology_name) -> Path:
        scenario_path = Path(os.path.dirname(__file__)).parent / 'Results' / config.evaluation_scenario_name \
                        / topology
        if run is None and exp is None and topology == config.topology_name:
            return scenario_path
        elif topology != config.topology_name and run is not None:
            scenario_path = scenario_path / run
        elif topology != config.topology_name and run is None:
            return scenario_path
        elif exp is None:
            scenario_path = scenario_path / f'Run_{run}'
        else:
            scenario_path = scenario_path / f'Run_{run}/{exp}'
        if not os.path.exists(scenario_path):
            if not os.path.exists(scenario_path.parent):
                os.mkdir(scenario_path.parent)
            os.mkdir(scenario_path)
        return scenario_path

    @staticmethod
    def clear_create_path(dir_path: Path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return
        if os.name == "nt":
            files = glob.glob(str(dir_path) + '\\*')
            for f in files:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
        else:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            os.mkdir(dir_path)

    @staticmethod
    @lru_cache(maxsize=100000)
    def parse_variable(variable, online=False) -> ():
        import re
        len_var = len(variable.split(','))
        if len_var == 3:
            raise NotImplementedError("Implement with re")
        elif len_var == 5:
            if online:
                raise NotImplementedError("Implement with re")
            else:
                node_pattern = re.compile(r'node\[\(([^,]+),([^,]+),([^,]+)\),([^,]+),([^,]+)\]')
                node_match = node_pattern.match(variable)
                if node_match:
                    app_name, alt, assoc_node, i, m = node_match.groups()
                    return app_name, alt, assoc_node, i, i, m, m
                else:
                    raise ValueError("The input string does not match the expected node format")
        elif len_var == 6:
            raise NotImplementedError("Implement with re")
        elif len_var == 7:
            pattern = re.compile(r'link\[\(([^,]+),([^,]+),([^,]+)\),([^,]+),([^,]+),([^,]+),([^,]+)\]')
            match = pattern.match(variable)
            if match:
                app_name, alt, assoc_node, i, j, m, n = match.groups()
                return app_name, alt, assoc_node, i, j, m, n
            else:
                raise ValueError("The input string does not match the expected format")
        elif len_var == 8:
            raise NotImplementedError("Implement with re")



