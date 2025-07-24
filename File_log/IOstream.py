
"""
@Time ： 2024/10/4 17:27
@Auth ： KECILIMU
@File ：IOstream.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
# Name: IOstream
# Time: 2024/9/21
# Auther: "Kecilimu"
# This file includes some functions for IO stream
import File_log.file_create as file_create
import torch
import csv
import pandas as pd
import pickle


def model_output(path, model):
    torch.save(model, path)
    return True


def convert_path_from_string_to_list(path_string):
    """
    Convert a path string to a list of paths.

    Args:
        path_string (str): The path string to be converted.

    Returns:
        list: The list of paths.
    """
    return [path_string]


def batch_convert_path_from_string_to_list(paths_string):
    """
    Convert a list of path strings to a list of lists of paths.

    Args:
        paths_string (list): The list of path strings to be converted.

    Returns:
        list: The list of lists of paths.
    """
    paths = []
    for path_string in paths_string:
        paths.append(convert_path_from_string_to_list(path_string))
    return paths


def save_list_to_csv(data, file_path):
    """
    Saves a list to a CSV file.

    Args:
        data (list): The list containing the data_origin to be saved.
        file_name (str): The name of the CSV file to be saved.
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
        file.flush()
        file.close()
    return True


def load_list_from_csv(file_path, header=None, sep=','):
    """
    Loads a list from a CSV file.

    Args:
        file_path (str): The name of the CSV file to be loaded.
        header (int): The row number to use as the column names.
        sep: the separator of the csv file

    Returns:
        list: The list loaded from the CSV file.
    """
    file_name = file_create.solo_path_check(file_path)
    data = pd.read_csv(file_name, header=header, sep=sep).values.tolist()
    return data


def load_list_from_excel(file_path):
    """
    Loads a list from a CSV file.

    Args:
        file_path (str): The name of the CSV file to be loaded.
        header (int): The row number to use as the column names.
        sep: the separator of the csv file

    Returns:
        list: The list loaded from the CSV file.
    """
    file_name = file_create.solo_path_check(file_path)
    data = pd.read_excel(file_name)
    return data.values.tolist()


def batch_load_list_from_csv(files, sep=','):
    """
    Loads list from CSV files.

    Args:
        files (list): The name of the CSV file to be loaded.

    Returns:
        list: The list loaded from the CSV file.
        sep: the separator of the csv file
    """
    datas = []
    for file_path in files:
        data = load_list_from_csv(file_path, sep=sep)
        datas.append(data)
    return datas


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return True


def load(path):
    if file_create.check_file_path(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    else:
        return False

def img_save(fig, path):
    fig.savefig(path)
    return True
