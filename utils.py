import csv
import json
import logging
import os
import sys


def list_to_csv(csv_fname: str, list_to_save: list):
    with open(csv_fname, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in list_to_save:
            writer.writerow([item])


def load_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def setup_logging(dataset_name):
    if not os.path.exists('.logfiles'):
        os.mkdir('.logfiles')
    logfile = f'.logfiles/{dataset_name}.log'
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=logfile,
        filemode='w'
    )

    # Create a handler for printing to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Attach the console handler to the root logger
    logging.getLogger('').addHandler(console_handler)

    # Redirect sys.stdout to both console and log file
    class DualLogger:
        def __init__(self, *writers):
            self.writers = writers

        def write(self, message):
            for writer in self.writers:
                writer.write(message)

        def flush(self):
            for writer in self.writers:
                writer.flush()

    sys.stdout = DualLogger(sys.stdout,
                            logging.getLogger('').handlers[0].stream)
