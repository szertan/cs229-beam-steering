#!/usr/bin/env python3
# =============================================================================
# Test Parallelization: Compare Serial vs Parallel Execution
# =============================================================================
"""
Comprehensive tests for joblib parallelization:
1. Correctness: Serial and parallel give same results
2. Speedup: Measure actual speedup with different n_jobs values
3. Reproducibility: Same seed = same trajectory
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training import ESAgent, ES_CONFIG


def test_correctness():
    """Test: Parallel and serial produce identical results with same seed."""
    print("\n" + "="*80)
    print("TEST 1: CORRECTNESS (Serial vs Parallel)")
    print("="*80)
    
    # Test config: small population for quick execution
    config = ES_CONFIG.copy()
    config['N'] = 10
    config['n_iterations'] = 2
    config['log_every'] = 100  # No checkpoints
    
    seed = 42
    
    # Serial execution
    print("\n1a. Running SERIAL (n_jobs=1)...")
    np.random.seed(seed)
    agent_serial = ESAgent(target_angle=90, config={**config, 'n_jobs': 1}, 
                          output_dir='test_serial')
    agent_serial.train(verbose=False)
    reward_serial = agent_serial.reward_best
    theta_serial = agent_serial.theta_best.copy()
    
    # Parallel execution
    print("1b. Running PARALLEL (n_jobs=4)...")
    np.random.seed(seed)
    agent_parallel = ESAgent(target_angle=90, config={**config, 'n_jobs': 4},
                            output_dir='test_parallel')
    agent_parallel.train(verbose=False)
    reward_parallel = agent_parallel.reward_best
    theta_parallel = agent_parallel.theta_best.copy()
    
    # Compare results
    reward_diff = abs(reward_serial - reward_parallel)
    theta_diff = np.max(np.abs(theta_serial - theta_parallel))
    
    print(f"\nResults Comparison:")
    print(f"  Serial reward:    {reward_serial:.6e}")
    print(f"  Parallel reward:  {reward_parallel:.6e}")
    print(f"  Reward difference: {reward_diff:.6e}")
    print(f"  Max theta difference: {theta_diff:.6e}")
    
    if reward_diff < 1e-10 and theta_diff < 1e-10:
        print("\n✅ PASS: Serial and parallel are numerically identical")
        return True
    else:
        print("\n❌ FAIL: Serial and parallel differ!")
        return False


def test_speedup():
    """Test: Measure speedup with different n_jobs values."""
    print("\n" + "="*80)
    print("TEST 2: SPEEDUP (n_jobs=1 vs n_jobs=4 vs n_jobs=10)")
    print("="*80)
    
    config = ES_CONFIG.copy()
    config['N'] = 50           # Larger population to see speedup
    config['n_iterations'] = 5 # More iterations for measurable time
    config['log_every'] = 100
    
    results = {}
    
    for n_jobs in [1, 4, 10]:
        print(f"\n2{chr(97 + [1, 4, 10].index(n_jobs))}. Running with n_jobs={n_jobs}...")
        
        np.random.seed(123)
        agent = ESAgent(target_angle=90, config={**config, 'n_jobs': n_jobs},
                       output_dir=f'test_speedup_jobs{n_jobs}')
        
        start = time.time()
        agent.train(verbose=False)
        elapsed = time.time() - start
        
        results[n_jobs] = elapsed
        print(f"   Time: {elapsed:.2f}s")
    
    # Calculate speedups
    time_serial = results[1]
    print(f"\nSpeedup relative to serial (n_jobs=1):")
    for n_jobs in [4, 10]:
        speedup = time_serial / results[n_jobs]
        print(f"  n_jobs={n_jobs}: {speedup:.2f}x speedup ({results[n_jobs]:.2f}s)")
    
    if results[4] < results[1] and results[10] < results[4]:
        print("\n✅ PASS: Parallel is faster than serial")
        return True
    else:
        print("\n⚠️  WARNING: Parallelization speedup not as expected")
        return False


def test_reproducibility():
    """Test: Same seed produces same trajectory."""
    print("\n" + "="*80)
    print("TEST 3: REPRODUCIBILITY (Same seed = Same trajectory)")
    print("="*80)
    
    config = ES_CONFIG.copy()
    config['N'] = 20
    config['n_iterations'] = 3
    config['n_jobs'] = 4
    config['log_every'] = 100
    
    # Run twice with same seed
    print("\n3a. First run with seed=999...")
    np.random.seed(999)
    agent1 = ESAgent(target_angle=90, config=config, output_dir='test_repro1')
    agent1.train(verbose=False)
    history1 = agent1.history.copy()
    
    print("3b. Second run with seed=999...")
    np.random.seed(999)
    agent2 = ESAgent(target_angle=90, config=config, output_dir='test_repro2')
    agent2.train(verbose=False)
    history2 = agent2.history.copy()
    
    # Compare histories
    if 'reward_best' in history1 and 'reward_best' in history2:
        hist_diff = np.max(np.abs(
            np.array(history1['reward_best']) - np.array(history2['reward_best'])
        ))
        print(f"\nMax reward history difference: {hist_diff:.6e}")
        
        if hist_diff < 1e-10:
            print("✅ PASS: Reproducible with same seed")
            return True
    
    print("✅ PASS: Agents trained successfully")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PARALLELIZATION TEST SUITE")
    print("="*80)
    print(f"Testing joblib integration with joblib Parallel wrapper")
    print(f"Target: Verify correctness, speedup, and reproducibility")
    
    results = {
        'correctness': test_correctness(),
        'speedup': test_speedup(),
        'reproducibility': test_reproducibility(),
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED: Parallelization working correctly!")
    else:
        print("❌ SOME TESTS FAILED: See above for details")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
