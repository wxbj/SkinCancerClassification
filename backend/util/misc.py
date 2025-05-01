import random
import numpy as np
import torch
import os
import logging
import sys

# 日志记录器
def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    # 创建日志父节点
    logger = logging.getLogger(name)
    # 设置父节点日记级别为最低，保证所有的信息都可以输出
    logger.setLevel(logging.DEBUG)
    # 保证只有0号进程会进行日志记录
    if distributed_rank > 0:
        return logger
    # 设置控制台日志输出格式
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 如果给定了输出文件夹，则设置日志输出到文件夹中
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# 固定随机种子
def set_random_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 保存检查点
def save_checkpoint(cfg,state, is_best):
    """将检查点保存到磁盘"""
    if is_best:
        directory = f"{cfg.OUTPUT_DIR}/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + f'/model_best1.pt'
        torch.save(state, filename)