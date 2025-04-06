import logging
import os
import datetime

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = (
        f"{log_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also log to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream_handler)

# Get a shared logger instance
logger = logging.getLogger(__name__)  # Each module gets its own logger
