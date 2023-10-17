"""
Benchmark an arbitrary console command by executing it a certain number of times.

Example: python benchmark.py python prodigy/cli.py --engine ginac main pgfexamples/inference/piranha.pgcl
"""

import subprocess
import sys

from prodigy.util.color import Style

count = 20
timeout = 90

times = []
for i in range(count):
    print(f"\ron iteration {i + 1}/{count}...", end="")
    # Choose timing behavior by commenting out
    psi_timings = True
    # psi_timings = False
    if psi_timings:
        result = subprocess.check_output(sys.argv[1:], timeout=timeout)
        if "seconds" not in result.decode().splitlines()[-2]:
            times.append(float(result.decode().splitlines()[-3].split()[0]))
        else:
            times.append(float(result.decode().splitlines()[-2].split()[0]))
    else:
        result = subprocess.check_output(sys.argv[1:], timeout=timeout)
        times.append(float(result.decode().splitlines()[-1].split()[2]))
print("\r", end="")

print(f"{Style.OKCYAN}min:{Style.RESET} {min(times):.3f} seconds")
print(f"{Style.OKCYAN}max:{Style.RESET} {max(times):.3f} seconds")
print(f"{Style.OKCYAN}average:{Style.RESET} {(sum(times) / count):.3f} seconds")

print()
print(f"{Style.OKCYAN}Output:{Style.RESET}\n{result.decode()}")
