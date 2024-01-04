import subprocess
import os
from prodigy.util.color import Style
from prodigy.util.logger import print_progress_bar


def benchmark(iterations, engine, path, limit):
    for filename in sorted(os.listdir(path)):

        # Skip invariant files.
        if "invariant" in filename:
            continue

        print(f"{Style.OKYELLOW}Currently benchmarking {Style.MAGENTA}{filename}{Style.RESET}".center(125, "-"))

        # Mention all exceptions so the user is informed.
        if filename == "digitRecognition.pgcl":
            print(f"{Style.OKRED}-> We give additional time for parsing. (this is not counted though, but the file is very large.){Style.RESET}")
            if engine == "sympy":
                print(f"{Style.OKRED}-> This benchmark might produce an error (Exceeding integer conversion size){Style.RESET}")

        elif filename == "lucky_throw.pgcl" and engine == "sympy":
            print(f"{Style.OKRED}-> This benchmark might produce an error (Maximum recursion limit){Style.RESET}")

        # Setup testing environment
        max_time = 0
        avg_time = 0
        min_time = 100000
        timeout=False
        is_parametrized = False

        # Do the actual benchmarking
        for i in range(iterations):

            # This is the echo process piping the invariant file in case of loopy programs
            prog_input_process = subprocess.Popen(
                ["echo", "-e", f"1\n{path}{filename.removesuffix('.pgcl')}_invariant.pgcl"],
                stdout=subprocess.PIPE,
                text=True
            )
            # this is the actual process to benchmark.
            timing_process = subprocess.Popen(
                ["python", "prodigy/cli.py", "--engine", engine, "main", f"{path}{filename}"],
                stdin=prog_input_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, 
                text=True
            )
            print_progress_bar(i, iterations, 1, 84 if i/iterations < 0.1 else 83, "\r")

            try:
                # if we encounter digitRecognition we give aditional time for parsing (file is huge).
                waiting_time = limit+90 if filename == "digitRecognition.pgcl" else limit+5

                # Do the job.
                output, error = timing_process.communicate(timeout=waiting_time)
                
                # Check wether the job was successfull (loopy output case)
                if "successfully validated" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                # Or the job presented aprameter valuations.
                elif "validated under" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                    if filename == "15_brp_obs_parameter.pgcl":
                        params = output.split("\n")[-5].split("[")[-1].removesuffix("]")
                    else:
                        params = output.split("\n")[-4].split("[")[-1].removesuffix("]")
                    is_parametrized = True

                # Plain programs:
                elif "seconds" in output:
                    current_time = float(output.split("\n")[-2].split()[-2])
                
                elif error is not None:
                    print(error.split("\n")[-1])
                    # If this happends, something fucked up.
                    print(f"{Style.OKRED}Skipping benchmark as error occured{Style.RESET}")
                    timeout = True
                    break
                
                # here we do the actual timeout check. 
                if current_time > limit:
                    timeout = True
                    print(" "*150, end='\r')
                    print(f"{Style.OKRED}Timeout{Style.RESET}")
                    break
                
                # Do the statistics
                max_time = max(max_time, current_time)
                min_time = min(min_time, current_time)
                avg_time += current_time

            except subprocess.TimeoutExpired:  # In case we got a timeout
                # cleanup and inform user.
                print(" "*150, end='\r')
                print(f"{Style.OKRED}Timeout{Style.RESET}")
                timing_process.kill()
                timeout=True
                break
        avg_time /= iterations

        # Print the results
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
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the GiNaC engine){Style.RESET}'.center(120, '='))
    print()
    benchmark(iterations, "ginac", "/root/artifact/pgfexamples/Table 4/", timeout)
    print(f'{Style.OKCYAN}Inference of Loopy Programs (Table 4 using the SymPy engine){Style.RESET}'.center(120, '='))
    print()
    benchmark(iterations, "sympy", "/root/artifact/pgfexamples/Table 4/", timeout)



def reproduce_loop_free(iterations, timeout):
    print()
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the GiNaC engine){Style.RESET}'.center(120, '='))
    print()
    benchmark(iterations, "ginac", "/root/artifact/pgfexamples/Appendix/", timeout)
    print(f'{Style.OKCYAN}Inference of Loop-free Programs (Appendix Table 5 using the SympY engine){Style.RESET}'.center(120, '='))
    print()
    benchmark(iterations, "sympy", "/root/artifact/pgfexamples/Appendix/", timeout)

def reproduce_all(iterations, timeout):
    reproduce_loopy(iterations, timeout)
    reproduce_loop_free(iterations, timeout)


if __name__ == '__main__':
    subprocess.run(["clear"])
    print(f'{Style.OKGREEN}REPLICATING RESULTS FROM THE OOPSLA 2024 PAPER{Style.RESET}'.center(120, "#"))
    choice = None
    while True:
        print()
        choice = input(f"Reproduce [l]oopy, [f] loop-free, [a]ll results or [q]uit? ")
        if choice.lower() in ['a', 'l', 'f']:
            break
        if choice.lower() == 'q':
            exit()

    iters = int(os.environ.get("ITERATIONS", 20))
    timeout = int(os.environ.get("TIMEOUT", 90))

    calls= {'a': reproduce_all, 'l':reproduce_loopy, 'f':reproduce_loop_free}
    print()
    print(f"{Style.YELLOW}Settings{Style.RESET}".center(120, "="))
    print(f"{Style.YELLOW}ITERATIONS={iters}\nTIMEOUT={timeout} seconds{Style.RESET}")
    print(f"{Style.YELLOW}Note: Every call gets additional time to compensate for python startup and parsing (this is not counted though){Style.RESET}")
    subprocess.run(["zsh", "-c", "sleep 5"])
    calls[choice](iters, timeout)

