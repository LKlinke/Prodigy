import subprocess
import os
from prodigy.util.color import Style
from prodigy.util.logger import print_progress_bar


def benchmark(iterations, engine, path, limit):
    for filename in sorted(os.listdir(path)):
        if "invariant" in filename:
            continue
        print(f"{Style.OKYELLOW}Currently benchmarking {filename}{Style.RESET}".center(120, "-"))

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
                    print("Timeout")
                    break
                max_time = max(max_time, current_time)
                min_time = min(min_time, current_time)
                avg_time += current_time
            except subprocess.TimeoutExpired:
                print("Timeout")
                timing_process.kill()
                timeout=True
                break
        avg_time /= iterations
        if not timeout:
            print(" "*150, end='\r')
            print(f"{Style.OKMAGENTA}Min time: {min_time:.3f} seconds{Style.RESET}")
            print(f"{Style.OKMAGENTA}Max time: {max_time:.3f} seconds{Style.RESET}")
            print(f"{Style.OKMAGENTA}Average time: {avg_time:.3f} seconds{Style.RESET}")
            if is_parametrized:
                print(f"{Style.OKMAGENTA}Parameter valuation: {params}{Style.RESET}")
        print()
        print()

def reproduce_loopy():
    print()
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the GiNaC engine){Style.RESET}'.center(120, '-'))
    benchmark(20, "ginac", "/root/artifact/pgfexamples/Table 4/", 90)
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the SymPy engine){Style.RESET}'.center(120, '-'))
    benchmark(20, "sympy", "/root/artifact/pgfexamples/Table 4/", 90)

def reproduce_loop_free():
    print()
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the GiNaC engine){Style.RESET}'.center(120, '-'))
    benchmark(20, "ginac", "/root/artifact/pgfexamples/Appendix/", 90)
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the SympY engine){Style.RESET}'.center(120, '-'))
    benchmark(20, "sympy", "/root/artifact/pgfexamples/Appendix/", 90)

def reproduce_all():
    print()
    print(f'{Style.OKGREEN}REPLICATING RESULTS FROM THE OOPSLA 2024 PAPER{Style.RESET}'.center(120, "-"))
    reproduce_loopy()
    reproduce_loop_free()


if __name__ == '__main__':
    choice = None
    while True:
        choice = input("Reproduce [l]oopy, [f] loop-free, [a]ll results or [q]uit?\t")
        if choice.lower() in ['a', 'l', 'f', 'q']:
            break
        print("Unrecognized input. Try Again.")

    calls= {'a': reproduce_all, 'l':reproduce_loopy, 'f':reproduce_loop_free , 'q':exit}
    calls[choice]()
