import logging
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """Set up the logger."""
    logger = logging.getLogger(__name__)  # 获取 logger 实例，如果参数为空返回 root logger
    logger.setLevel(log_level)  # 设置日志记录级别
    
    # 定义日志输出格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    
    # 输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        # FileHandler，写入日志文件
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger