import logging
import time


def set_logger(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{args.name}.{int(time.time())}"),
            logging.StreamHandler()
        ]
    )
