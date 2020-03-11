"""
This script implements all tool functions.
"""
import json
import logging
import yaml
import time
import datetime


def __logger():
    """
    Define logger.
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def _read(path):
    """
    Read json by lines.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    logger = __logger()
    logger.info("Read json by lines, from path: {}.".format(path))
    return lines


def __read(path):
    """
    Read the whole json.
    """
    with open(path, "r") as f:
        data = json.load(f)
    logger = __logger()
    logger.info("Read the whole json, from path: {}.".format(path))
    return data


def _save(path, data):
    """
    Save data to json by lines.
    """
    with open(path, 'w') as fout:
        for item in data:  # [{}, {}, ...]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger = __logger()
    logger.info("Save data to json by lines, to: {}.".format(path))


def __save(path, data):
    """
    Save the whole json.
    """
    with open(path, 'w') as fout:
        fout.write(json.dumps(data, ensure_ascii=False))
    logger = __logger()
    logger.info("Save the whole json, to: {}.".format(path))


def load_config():
    """
    From yaml load configuration.
    """
    path = "config/model_config.yaml"
    with open(path, 'r', encoding="utf-8") as f:
        config_data = f.read()
    config = yaml.load(config_data)  # dict
    logger = __logger()
    logger.info("From {} load configuration.".format(path))
    return config


def get_time():
    """
    Get current time.
    """
    format_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    now_time = datetime.datetime.now()
    logger = __logger()
    logger.info("Get format_time and now_time.")
    return format_time, now_time


def cost_time(start_time, end_time):
    """
    获取时间差 (格式化)
    :param start_time: format_time
    :param end_time: format_time
    :return: format_time
    """
    _cost = end_time - start_time
    return _cost

