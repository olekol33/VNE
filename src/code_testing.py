import cProfile as profile
from include import config
import contextlib
import time
from pathlib import Path


@contextlib.contextmanager
def timer(block_name: str):
    if not config.measure_function_runtime:
        yield
        return
    start = time.time()
    yield
    end = time.time()
    total_time = round(end - start, 2)
    print(f'{block_name} executed in {total_time} seconds.')
