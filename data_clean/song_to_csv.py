import os
import csv
from mutagen.id3 import ID3
from flask import Flask, jsonify, request

app = Flask(__name__)

# 指定包含MP3文件的目录路径
#mp3_dir = "E:/music/华语歌曲"

# 指定输出CSV文件的路径
#output_csv_path = "../data/mp3_to_csv.csv"

def parse_id3_tags(file_path):
    try:
        audio = ID3(file_path)
        tags = {
            'title': audio.get('TIT2', [''])[0],
            'artist': audio.get('TPE1', [''])[0],
            'album': audio.get('TALB', [''])[0],
            'track_number': audio.get('TRCK', [''])[0],
            'year': audio.get('TYER', [''])[0] or audio.get('TDRC', [''])[0],
            'genre': audio.get('TCON', [''])[0],
            'custom_id': None
        }

        # 查找描述为“Custom ID”的TXXX帧
        for frame in audio.getall('TXXX'):
            if frame.desc == 'Custom ID':
                tags['custom_id'] = frame.text[0]

        return tags
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_mp3_info_to_csv(input_dir, output_csv_path):
    mp3_dir = input_dir
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'title',
            'artist',
            'album',
            'track_number',
            'year',
            'genre',
            'custom_id'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for root, dirs, files in os.walk(mp3_dir):
            for file in files:
                if file.endswith('.mp3'):
                    file_path = os.path.join(root, file)
                    tags = parse_id3_tags(file_path)
                    if tags is not None:
                        writer.writerow(tags)

@app.route('/extract_mp3_info', methods=['POST'])
def extract_mp3_info():
    # 接收请求体中的MP3文件目录路径和输出CSV文件路径
    input_dir = request.json.get('input_dir')
    output_csv_path = request.json.get('output_csv_path', '../data/mp3_to_csv.csv')

    if input_dir is None:
        return jsonify({"status": "error", "message": "Missing required parameter 'input_dir'."}), 400

    extract_mp3_info_to_csv(input_dir, output_csv_path)

    return jsonify({"status": "success", "message": f"MP3 info extracted from {input_dir} to {output_csv_path}."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)