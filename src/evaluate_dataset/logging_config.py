import logging
import os
import datetime

def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = (
        f"{log_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    # Don't show logs from the litellm library
    logging.getLogger("litellm").setLevel(logging.ERROR)

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
