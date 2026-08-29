# memory_profiler.py
import torch
import gc
import time
from contextlib import contextmanager

class MemoryProfiler:
    """
    Memory profiler for measuring peak GPU memory usage.
    """
    def __init__(self, device='cuda:0'):
        self.device = device
        self.is_cuda = device != 'cpu' and 'cuda' in str(device)
        self.reset()
        
    def reset(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_memory = torch.cuda.memory_allocated(self.device)
        else:
            self.start_memory = 0
        
        self.start_time = time.time()
        self.checkpoints = []
        self.peak_memory = 0
        
    def checkpoint(self, name):
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
            current_memory = torch.cuda.memory_allocated(self.device)
            peak_memory = torch.cuda.max_memory_allocated(self.device)
        else:
            import psutil
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            peak_memory = current_memory
        
        elapsed = time.time() - self.start_time
        
        self.checkpoints.append({
            'name': name,
            'current_mb': current_memory,
            'peak_mb': peak_memory,
            'elapsed': elapsed
        })
        
        if peak_memory > self.peak_memory:
            self.peak_memory = peak_memory
            
    def report(self):
        print("\n" + "=" * 70)
        print("MEMORY PROFILING REPORT".center(70))
        print("=" * 70)
        
        for cp in self.checkpoints:
            print(f"{cp['name']}:")
            print(f"  Current: {cp['current_mb']:.2f} MB")
            print(f"  Peak:    {cp['peak_mb']:.2f} MB")
            print(f"  Time:    {cp['elapsed']:.2f}s")
            print()
        
        if len(self.checkpoints) > 1:
            start = self.checkpoints[0]['current_mb']
            peak = max(cp['peak_mb'] for cp in self.checkpoints)
            end = self.checkpoints[-1]['current_mb']
            print("-" * 70)
            print(f"Start Memory: {start:.2f} MB")
            print(f"Peak Memory:  {peak:.2f} MB")
            print(f"End Memory:   {end:.2f} MB")
            print(f"Memory Used:  {peak - start:.2f} MB")
            print("=" * 70)
        
        return peak - start if len(self.checkpoints) > 1 else 0

    
@contextmanager
def profile_memory(name, profiler=None):
    if profiler is None:
        profiler = MemoryProfiler()
    try:
        yield profiler
    finally:
        if name:
            profiler.checkpoint(name)