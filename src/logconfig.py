import logging
import os
import sys

class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Provide defaults for custom fields if missing
        for field in ['job_id', 'nodename', 'rank', 'local_rank']:
            if not hasattr(record, field):
                setattr(record, field, 'N/A')
        return super().format(record)

class CustomFilter(logging.Filter):
    """
    A custom filter class for logging that will be used to prepend the rank, local_rank and nodename to the log messages.
    """
    def __init__(self, job_id, rank, local_rank, nodename):
        super().__init__()
        self.job_id = job_id
        self.rank = rank
        self.local_rank = local_rank
        self.nodename = nodename

    def filter(self, record):
        record.job_id = self.job_id
        record.rank = self.rank
        record.local_rank = self.local_rank
        record.nodename = self.nodename
        return True

def setup_logging(console_level=logging.INFO):
    """
    Set up root logger with a console handler and configure
    warnings capture and uncaught exception logging.
    """
    formatter = SafeFormatter('%(job_id)s | %(nodename)s | @rank: %(rank)s | @local: %(local_rank)s at %(asctime)s | %(levelname)s | %(name)s  | \t %(message)s')

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    logging.captureWarnings(True)

    # Setup uncaught exception logging
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let keyboard interrupts pass through without logging
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Log to the main logger only - it will appear in all log files due to file handlers
        main_logger = logging.getLogger("__main__")
        if main_logger.handlers:
            main_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        else:
            # Fallback to root logger if main logger has no handlers
            root_logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_uncaught_exception

def add_file_logger(logger_name, file_path, level=logging.DEBUG, rank=0, local_rank=0):
    """
    Add file handler to specific logger.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    logger = logging.getLogger(logger_name)
    
    # Check if this specific logger already has a FileHandler for this file
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(file_path):
            return  # FileHandler for this file already exists
    
    logger.setLevel(logging.DEBUG)
    job_id = ""
    if "SLURM_JOB_ID" in os.environ:
        job_id = f'job:{os.environ["SLURM_JOB_ID"]}'
    file_handler = logging.FileHandler(file_path, mode='a')
    formatter = SafeFormatter('%(job_id)s | %(nodename)s | @rank: %(rank)s | @local: %(local_rank)s at %(asctime)s | %(levelname)s | %(name)s  | \t %(message)s')
    custom_filter = CustomFilter(job_id, rank, local_rank, os.uname().nodename)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(custom_filter)
    file_handler.setLevel(level)

    logger.addHandler(file_handler)

    # Propagate True = still outputs to console (via root logger)
    logger.propagate = True