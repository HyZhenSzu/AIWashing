import time

class Timer:
    def __init__(self, name: str = None):
        self.name = name
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError(f"Timer '{self.name}' was not started")
        elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop()
        name = f" '{self.name}'" if self.name else ""
        print(f"Timer{name}: {elapsed:.4f} seconds")
