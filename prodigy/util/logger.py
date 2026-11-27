import logging
import os
from typing import Optional, Union

from prodigy.util.color import Style


LOG_FILE_ENV = "PRODIGY_LOG_FILE"


def log_setup(name: str, level: Union[int, str], file: Optional[str] = None) -> logging.Logger:
    """
    The logger ``name``. By default it writes nothing to disk and its level is WARNING: records of level WARNING
    and above reach stderr through Python's last-resort handler, and a debug or info call creates no record at all
    (its arguments are formatted only for a record, so the calls stay cheap). DEBUG output (it contains whole
    generating functions) is written only on request, to ``file`` or to the file named by the environment variable
    ``PRODIGY_LOG_FILE``; then the logger has the given ``level`` (#284).
    """
    logger = logging.getLogger(name)
    file = file or os.environ.get(LOG_FILE_ENV)
    numeric = level if isinstance(level, int) else logging.getLevelNamesMapping()[level]
    logger.setLevel(numeric if file else max(numeric, logging.WARNING))
    if file and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(file)
                        for h in logger.handlers):
        fhandler = logging.FileHandler(filename=file, mode='a')
        fhandler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fhandler)
    return logger


# Print iterations progress
def print_progress_bar(iteration: int, total: int, decimals: int = 1, length: int = 100, print_end: str = "") -> None:
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
