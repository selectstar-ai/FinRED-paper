import logging
import sys
from pathlib import Path
from typing import Optional

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "application.log"

def get_logger(name: str, log_file: Optional[str] = str(DEFAULT_LOG_FILE), level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # 중복 핸들러 방지
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택적)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger