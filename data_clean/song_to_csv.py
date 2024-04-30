import os
import csv
from mutagen.id3 import ID3
from mutagen.flac import FLAC
from flask import Flask, jsonify, request

app = Flask(__name__)

def parse_audio_tags(file_path):
    try:
        audio = None
        if file_path.endswith('.mp3'):
            audio = ID3(file_path)
        elif file_path.endswith('.flac'):
            audio = FLAC(file_path)
        else:
            print(f"Unsupported file format for {file_path}")
            return None

        tags = {
            'title': audio.get('TIT2', [''])[0] if isinstance(audio, ID3) else audio['title'][0],
            'artist': audio.get('TPE1', [''])[0] if isinstance(audio, ID3) else audio['artist'][0],
            'album': audio.get('TALB', [''])[0] if isinstance(audio, ID3) else audio['album'][0],
            'track_number': audio.get('TRCK', [''])[0] if isinstance(audio, ID3) else audio['tracknumber'][0],
            'year': audio.get('TYER', [''])[0] or audio.get('TDRC', [''])[0] if isinstance(audio, ID3) else audio['date'][0],
            'genre': audio.get('TCON', [''])[0] if isinstance(audio, ID3) else audio['genre'][0],
            'custom_id': None
        }

        # 查找描述为“Custom ID”的TXXX帧（仅适用于MP3）
        if isinstance(audio, ID3):
            for frame in audio.getall('TXXX'):
                if frame.desc == 'Custom ID':
                    tags['custom_id'] = frame.text[0]
        # 获取FLAC文件中的自定义comments标签
        elif isinstance(audio, FLAC):
            comments = audio.comments.get('COMMENT', [])
            if len(comments) > 0:
                tags['custom_comments'] = comments[0]

        return tags
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def extract_audio_info_to_csv(input_dir, output_csv_path):
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

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.mp3') or file.endswith('.flac'):
                    file_path = os.path.join(root, file)
                    tags = parse_audio_tags(file_path)
                    if tags is not None:
                        writer.writerow(tags)

