
"""
@Time ： 2024/10/4 17:39
@Auth ： KECILIMU
@File ：file_create.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# Name: file_create
# Time: 2024/9/21
# Auther: "Kecilimu"

import os
import datetime
import time

import torch

import utils

def reflash_time():
    return datetime.datetime.now()


def init_time():
    time_now = reflash_time().strftime("%Y%m%d_%H%M%S")
    return time_now


def random_seed():
    # use datetime now to generate a random seed
    return int(reflash_time().timestamp())


def create_log_path(file_name, result_path=utils.result_path):
    log_file = result_path + '/' + file_name + '.txt'
    os.makedirs(result_path, exist_ok=True)
    return log_file


def create_model_path(file_name, result_path=utils.result_path):
    model_file = result_path + '/' + file_name + '/model.pth'
    os.makedirs(result_path, exist_ok=True)
    return model_file


def create_result_path(file_name, result_path=utils.result_path):
    result_file = result_path + '/' + file_name
    os.makedirs(result_path, exist_ok=True)
    return result_file


def check_dictory_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return True


def check_file_path(file_path):
    if not os.path.exists(file_path):
        return False
    return True


def note(file_path, file_content=None):
    # Check if the file exists
    with open(file_path, 'a') as file:
        file.write(file_content + "\n\t" + f"File '{file_path}' noted successfully.\n")
    return True


def convert_path_in_main(file):
    """
    Convert the path in main.py to the path in default_parameter.py

    Args:
        file : The path to be converted.

    Returns:
        new_files: The converted path.
    """
    file = solo_path_check(file)
    return file.replace("../", "")


def batch_convert_path_in_main(files):
    """
    Convert the path in main.py to the path in default_parameter.py

    Args:
        files : The paths to be converted.

    Returns:
        new_files: The converted paths.
    """
    new_files = []
    for file in files:
        new_file = convert_path_in_main(file)
        new_files.append(new_file)
    return new_files

def solo_path_check(path):
    """
    Check if the path is a string or a list.

    Args:
        path : The path to be checked.

    Returns:
        path : The path.
    """
    if isinstance(path, list):
        return path[0]
    else:
        return path


def torch_seed_set():
    """
    Set the seed for torch.

    Returns:
        seed : The seed for torch.
    """
    seed = int(time.time())
    torch.manual_seed(seed)
    return seed