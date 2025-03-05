import time

class Timer:
    def __init__(self, label="Execution"):
        self.label = label

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        print(f"{self.label} took {elapsed_time:.6f} seconds")