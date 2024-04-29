import csv
import json


def list_to_csv(csv_fname: str, list_to_save: list):
    with open(csv_fname, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in list_to_save:
            writer.writerow([item])


def load_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
