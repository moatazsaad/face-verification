import time  
import json  
import argparse  
import subprocess  
import sys
from concurrent.futures import ThreadPoolExecutor  # used to run requests in parallel


# Compute p95 latency
def p95(values):
    if not values:
        return 0
    values = sorted(values)
    return values[int(0.95 * (len(values) - 1))]


# Run ONE inference request using your CLI
# This simulates a real user calling the system from terminal
def run_one(image1, image2):
    start = time.perf_counter()  # start timer 

    # Call your CLI as a user 
    result = subprocess.run(
        [
            sys.executable,
            "-m","src.run_inference_cli",
            "--image1",
            image1,
            "--image2",
            image2,
        ],
        capture_output=True,
        text=True,
    )

    # Compute latency in milliseconds
    latency = (time.perf_counter() - start) * 1000

    # Return 0/1 for succeed/fail and latency
    return result.returncode, result.stderr, latency


def main():
    # Create tool that reads command line arguments for load test
    parser = argparse.ArgumentParser(description="Simple concurrent load test")

    # Path to file with fixed image pairs 
    parser.add_argument("--pairs_file", required=True)

    # Total number of requests to run
    parser.add_argument("--requests", type=int, default=10)

    # Number of parallel workers
    parser.add_argument("--workers", type=int, default=2)
    
    # Read inputs from terminal
    args = parser.parse_args()

    # Load deterministic set of image pairs
    with open(args.pairs_file, "r") as f:
        pairs = json.load(f)

    latencies = []  
    failures = 0    

    start_time = time.perf_counter()

    # Run requests in parallel using a thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []

        # Submit all requests
        for i in range(args.requests):
            # Cycle through pairs so we reuse the fixed set
            pair = pairs[i % len(pairs)]

            # Submit one request to the worker pool
            futures.append(
                executor.submit(run_one, pair["image1"], pair["image2"])
            )

        # Collect results after execution
        for f in futures:
            code, stderr, latency = f.result()

            if code == 0:
                latencies.append(latency)  # success → record latency
            else:
                failures += 1  # failure → count it
                print(stderr)

    total_time = time.perf_counter() - start_time

    # Throughput = successful requests per second
    throughput = len(latencies) / total_time if total_time > 0 else 0

    # Print final results
    print("\nSummary")
    print("--------")
    print("total_requests:", args.requests)  
    print("success:", len(latencies))        
    print("failures:", failures)            
    print("total_wall_time_sec:", total_time)  
    print("throughput (req/sec):", throughput)  
    print("avg_latency_ms:", sum(latencies) / len(latencies) if latencies else 0)  
    print("max_latency_ms:", max(latencies) if latencies else 0)  
    print("p95_latency_ms:", p95(latencies))  


if __name__ == "__main__":
    main()
    # python -m scripts.run_load_test --pairs_file examples/load_test_pairs.json --requests 10 --workers 3