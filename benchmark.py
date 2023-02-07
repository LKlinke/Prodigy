"""
Benchmark an arbitrary console command by executing it a certain number of times.

Example: python benchmark.py python prodigy/cli.py --engine ginac main pgfexamples/inference/piranha.pgcl
"""

import subprocess
import sys
import time

from prodigy.util.color import Style

count = 20
timeout = 30

times = []
for i in range(count):
    print(f"\ron iteration {i+1}/{count}...", end="")
    start = time.perf_counter()
    result = subprocess.check_output(sys.argv[1:], timeout=timeout)
    times.append(time.perf_counter() - start)
print("\r", end="")

print(f"{Style.OKCYAN}min:{Style.RESET} {min(times)} seconds")
print(f"{Style.OKCYAN}max:{Style.RESET} {max(times)} seconds")
print(f"{Style.OKCYAN}average:{Style.RESET} {sum(times) / count} seconds")

print()
print(f"{Style.OKCYAN}Output:{Style.RESET}\n{result.decode()}")
