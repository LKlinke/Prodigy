import subprocess
import os
from prodigy.util.color import Style
from prodigy.util.logger import print_progress_bar


def benchmark(iterations, engine, path, limit):
    for filename in sorted(os.listdir(path)):
        if "invariant" in filename:
            continue
        print(f"{Style.OKYELLOW}Currently benchmarking {filename}{Style.RESET}".center(120, "-"))
        if filename == "digitRecognition.pgcl" and engine == "sympy":
            print(f"{Style.OKRED}>This benchmark might produce an error (Exceeding integer conversion size){Style.RESET}".rjust(120, "-"))
        elif filename == "lucky_throw.pgcl" and engine == "sympy":
            print(f"{Style.OKRED}>This benchmark might produce an error (Maximum recursion limit){Style.RESET}".rjust(120, "-"))
        max_time = 0
        avg_time = 0
        min_time = 100000
        timeout=False
        is_parametrized = False
        for i in range(iterations):
            prog_input_process = subprocess.Popen(["echo", "-e", f"1\n{path}{filename.removesuffix('.pgcl')}_invariant.pgcl"], stdout=subprocess.PIPE, text=True)
            timing_process = subprocess.Popen(
                ["python", "prodigy/cli.py", "--engine", engine, "main", f"{path}{filename}"],
                stdin=prog_input_process.stdout,
                stdout=subprocess.PIPE, 
                text=True
            )
            print_progress_bar(i, iterations, 1, 84 if i/iterations < 0.1 else 83, "\r")
            try:
                output, error = timing_process.communicate(timeout=limit+100)
                # This is only for the loopy programs.
                if "successfully validated" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                elif "validated under" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                    if filename == "15_brp_obs_parameter_invariant.pgcl":
                        params = output.split("\n")[-5].split("[")[-1].removesuffix("]")
                    else:
                        params = output.split("\n")[-4].split("[")[-1].removesuffix("]")
                    is_parametrized = True
                # Plain programs:
                elif "seconds" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                else:
                    print("This is not intended to happen!")
                    exit()
                if current_time > limit:
                    timeout = True
                    print(" "*150, end='\r')
                    print(f"{Style.OKRED}Timeout{Style.RESET}")
                    break
                max_time = max(max_time, current_time)
                min_time = min(min_time, current_time)
                avg_time += current_time
            except subprocess.TimeoutExpired:
                print(" "*150, end='\r')
                print(f"{Style.OKRED}Timeout{Style.RESET}")
                timing_process.kill()
                timeout=True
                break
        avg_time /= iterations
        if not timeout:
            print(" "*150, end='\r')
            print(f"Min time: {min_time:.3f} seconds{Style.RESET}")
            print(f"Max time: {max_time:.3f} seconds{Style.RESET}")
            print(f"{Style.OKGREEN}Average time: {avg_time:.3f} seconds{Style.RESET}")
            if is_parametrized:
                print(f"{Style.OKGREEN}Parameter valuation: {params}{Style.RESET}")
        print()
        print()

def reproduce_loopy(iterations, timeout):
    print()
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the GiNaC engine){Style.RESET}'.center(120, '-'))
    benchmark(iterations, "ginac", "/root/artifact/pgfexamples/Table 4/", timeout)
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the SymPy engine){Style.RESET}'.center(120, '-'))
    benchmark(iterations, "sympy", "/root/artifact/pgfexamples/Table 4/", timeout)

def reproduce_loop_free(iterations, timeout):
    print()
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the GiNaC engine){Style.RESET}'.center(120, '-'))
    benchmark(iterations, "ginac", "/root/artifact/pgfexamples/Appendix/", timeout)
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the SympY engine){Style.RESET}'.center(120, '-'))
    benchmark(iterations, "sympy", "/root/artifact/pgfexamples/Appendix/", timeout)

def reproduce_all(iterations, timeout):
    print()
    print(f'{Style.OKGREEN}REPLICATING RESULTS FROM THE OOPSLA 2024 PAPER{Style.RESET}'.center(120, "-"))
    reproduce_loopy(iterations, timeout)
    reproduce_loop_free(iterations, timeout)


if __name__ == '__main__':
    choice = None
    while True:
        print()
        print()
        choice = input("Reproduce [l]oopy, [f] loop-free, [a]ll results or [q]uit?\t")
        if choice.lower() in ['a', 'l', 'f', 'q']:
            break
        if choice.lower() == 'q':
            exit()

    iters = int(os.environ.get("ITERATIONS", 20))
    timeout = int(os.environ.get("TIMEOUT", 90))

    calls= {'a': reproduce_all, 'l':reproduce_loopy, 'f':reproduce_loop_free}
    print(f"{Style.OKWHITE} Current Setting: {iters} iter. per benchmark; {timeout=} seconds{Style.RESET}")
    calls[choice](iters, timeout)
