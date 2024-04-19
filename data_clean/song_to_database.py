from flask import Flask, request, jsonify
import os
import shutil
import tinytag
from mutagen.mp3 import MP3

app = Flask(__name__)

def get_release_year(file_path, file_format):
    if file_format.lower() == '.mp3':
        audio = MP3(file_path)
        tags = audio.tags
        if 'TDRC' in tags:
            year = tags['TDRC'][0].text[0:4]
        elif 'TYER' in tags:
            year = tags['TYER'][0].text
        else:
            year = None
    elif file_format.lower() == '.flac':
        audio = tinytag.TinyTag.get(file_path)
        year = audio.year
    else:
        year = None

    return year

def organize_audio_files(input_dir, output_root_dir):
    for dirpath, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(dirpath, file)
            file_format = os.path.splitext(file)[1].lower()
            if file_format in ['.mp3', '.flac']:
                year = get_release_year(file_path, file_format)
                if year is not None:
                    dest_folder = os.path.join(output_root_dir, f'{year}年音乐')
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest_file_path = os.path.join(dest_folder, file)
                    shutil.move(file_path, dest_file_path)
                    print(f'Moved "{file}" to "{dest_folder}"')
@app.route('/songToDatabase', methods=['POST'])
def organize():
    data = request.get_json()

    input_dir = data.get('input_directory')
    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({'error': 'Invalid input directory'}), 400

    output_dir = data.get('output_directory')
    if not output_dir:
        return jsonify({'error': 'Missing output directory'}), 400

    try:
        organize_audio_files(input_dir,output_dir)
        return jsonify({'message': 'Audio files organized successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)