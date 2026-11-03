import logging

from prodigy.util.color import Style


def log_setup(name: str, level, file: str = 'test.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fhandler = logging.FileHandler(filename=file, mode='a')
    fhandler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fhandler)
    return logger


# Print iterations progress
def print_progress_bar(iteration, total, decimals=1, length=100, print_end=""):
    """
    Call in a loop to create terminal progress bar

    Args:
        iteration: current iteration
        total: total iterations
        decimals: positive number of decimals in percent complete
        length: character length of bar
        print_end: end character, e.g. a carriage return
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    progress_bar = '█' * filled_length + '-' * (length - filled_length)
    print(
        f'{Style.YELLOW if (iteration / float(total)) < 1 else Style.GREEN}\r' \
        f'Progress: |{progress_bar}| {percent}% completed{Style.RESET}',
        end=print_end,
        flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()
