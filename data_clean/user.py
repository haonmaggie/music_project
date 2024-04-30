from flask import Flask, request, jsonify
import secrets
import string
import csv
import os

def save_to_csv(data, out_dir, filename='user_ids.csv'):
    output_file_path = os.path.join(out_dir, filename)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id'])

        for user_id in data:
            writer.writerow([user_id])

