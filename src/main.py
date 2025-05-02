from file_handler import FileHandler
from include import config
from experiments import RunExperiment
import time
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    FileHandler.create_run_dir(clear=True)
    print(f"Seed used: {config.seed}")


    start_time = time.time()
    exp = RunExperiment()
    print(f"Total time: {time.time() - start_time}")



