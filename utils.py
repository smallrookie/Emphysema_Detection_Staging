import os
import random

import monai
import numpy as np
import torch


def config_cpu_num(cpu_num):
    """Configure the number of CPU threads for various libraries.
    
    This function sets environment variables to control the number of threads
    used by different numerical libraries to optimize performance.
    
    Args:
        cpu_num (int): Number of CPU threads to use
    """
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def set_random_seed(seed=42):
    """Set random seed for reproducibility.
    
    This function sets the random seed for Python's random module, NumPy,
    and PyTorch to ensure reproducible results.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # monai.utils.set_determinism(seed=seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import logging
import os
import sys


def setup_logging(log_file_name, save_path):
    """Set up logging configuration.
    
    This function configures the logging system to write logs to both a file
    and the console (stdout).
    
    Args:
        log_file_name (str): Name of the log file
        save_path (str): Directory path to save the log file
    """
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(save_path, log_file_name)
            ),  # File where logs will be saved
            logging.StreamHandler(sys.stdout),  # Also output to the console
        ],
    )


def log_config_details(args):
    """
    Logs all the configuration settings (arguments).
    
    This function logs all the configuration parameters used for training
    or testing to facilitate experiment tracking.
    
    Args:
        args (argparse.Namespace): Arguments containing configuration settings
    """
    logging.info("Logging training/testing configurations:")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")
    logging.info("------------------- end of configurations -------------------")