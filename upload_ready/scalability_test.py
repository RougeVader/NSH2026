import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acm.conjunction.screening import ConjunctionScreening
from acm.utils.benchmarking import update_scan_benchmark

def test_scalability():
    print("\n--- Testing Scalability (1000 objects, 24h scan) ---")
    screener = ConjunctionScreening()
    
    # 1000 debris objects in LEO
    N = 1000
    debris_states = np.random.rand(N, 6) * 100.0
    # Normalize to shells around 7000km
    for i in range(N):
        r = debris_states[i, :3]
        debris_states[i, :3] = (r / np.linalg.norm(r)) * (7000.0 + np.random.randn()*50)
        debris_states[i, 3:] = (np.random.rand(3) - 0.5) * 15.0 # High relative velocity
        
    sat_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0])
    
    start_time = time.time()
    candidates = screener.long_range_scan(sat_state, debris_states, t_start=1000.0, horizon_s=86400.0, dt_scan=600.0)
    end_time = time.time()
    
    exec_time = end_time - start_time
    print(f"Scan took {exec_time:.4f} seconds for {N} objects over 24h.")
    print(f"Found {len(candidates)} potential candidates.")
    
    # Update BENCHMARKS.md automatically
    update_scan_benchmark(exec_time)
    print("Automatically updated BENCHMARKS.md with new results.")
    
if __name__ == "__main__":
    test_scalability()
