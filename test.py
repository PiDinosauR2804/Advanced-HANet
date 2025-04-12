from loguru import logger
from tqdm import tqdm

# Xoá handler mặc định
logger.remove()

# Ghi log ra file (không cần màu)
logger.add("logs/run.log", level="DEBUG", enqueue=True)

logger.add(
    lambda msg: tqdm.write(msg, end=""),
    level="DEBUG",
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{file: >18}: {line: <4}</cyan> - <level>{message}</level>", 
)

logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning.")
logger.error("This is an error!")
logger.critical("CRITICAL error!")
