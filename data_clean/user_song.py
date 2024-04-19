import os
import random
import csv
import uuid
import numpy as np
from collections import Counter
from mutagen.id3 import ID3, TXXX
from mutagen.flac import FLAC, Picture
from flask import Flask, jsonify, request

app = Flask(__name__)

# 定义处理MP3/FLAC文件的函数
def process_audio_files(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename.endswith('.mp3'):
                process_mp3_file(filepath)
            elif filename.endswith('.flac'):
                process_flac_file(filepath)

def process_mp3_file(filepath):
    audio = ID3(filepath)

    # 生成类似UUID格式的编号
    unique_id = str(uuid.uuid4()).upper().replace('-', '')

    # 创建一个TXXX帧用于存储自定义编号
    custom_id_frame = TXXX(encoding=3, description='Custom ID', text=unique_id)

    # 添加或更新ID3标签中的自定义编号
    if 'TXXX:Custom ID' not in audio:
        audio.add(custom_id_frame)
    else:
        audio['TXXX:Custom ID'] = custom_id_frame

    # 保存更改
    audio.save(v2_version=3)  # 保存为ID3v2.3版本

def process_flac_file(filepath):
    audio = FLAC(filepath)

    # 生成类似UUID格式的编号
    unique_id = str(uuid.uuid4()).upper().replace('-', '')

    # 在VORBIS_COMMENT中添加自定义编号
    audio['comments'][unique_id.encode()] = b''

    # 保存更改
    audio.save()

# 遍历目录及其子目录下的所有MP3和FLAC文件
def extract_song_custom_ids(directory):
    songs = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".mp3") or filename.endswith(".flac"):
                try:
                    if filename.endswith(".mp3"):
                        audio = ID3(os.path.join(root, filename))
                    elif filename.endswith(".flac"):
                        audio = FLAC(os.path.join(root, filename))

                    custom_id = None
                    if filename.endswith(".mp3"):
                        for frame in audio.values():
                            if isinstance(frame, TXXX) and frame.desc == 'Custom ID':
                                custom_id = frame.text[0]
                                break
                    elif filename.endswith(".flac"):
                        for key, value in audio['comments'].items():
                            if key.decode() == value.decode():  # Compare key and value
                                custom_id = key.decode()
                                break

                    if custom_id is not None:
                        songs.append(custom_id)
                        print(f"Found custom ID '{custom_id}' in file '{os.path.join(root, filename)}'")
                    else:
                        print(f"No custom ID found in file '{os.path.join(root, filename)}'")
                except Exception as e:
                    print(f"Error reading audio tag from file {os.path.join(root, filename)}: {e}")

    return songs

# 从指定的CSV文件中读取用户ID列表
def read_users_from_csv(csv_file_path):
    users = []
    with open(csv_file_path, "r") as user_file:
        reader = csv.reader(user_file)
        for row in reader:
            users.extend(row)

    return users


def generate_play_records(users, songs, avg_plays_per_song, play_records_per_user):
    """
    Generate play records for each user based on the given parameters.

    Args:
        users (list[str]): List of user IDs.
        songs (list[str]): List of song custom IDs.
        avg_plays_per_song (float): Average number of plays per song.
        play_records_per_user (int): Number of play records to generate per user.

    Returns:
        list[tuple]: List of play records, where each record is a tuple containing (user_id, song_id).
    """
    play_records = []

    # Calculate the total number of plays needed across all users
    total_plays_needed = len(users) * play_records_per_user

    # Generate a Poisson distribution with the specified average plays per song
    poisson_distribution = np.random.poisson(avg_plays_per_song, size=len(songs))

    # Normalize the distribution so that it sums up to the total plays needed
    distribution_sum = sum(poisson_distribution)
    if distribution_sum != total_plays_needed:
        poisson_distribution *= total_plays_needed / distribution_sum

    # Convert the distribution to a list of tuples representing individual plays
    plays = [(song_id, count) for song_id, count in zip(songs, poisson_distribution)]

    # Distribute the plays randomly among users
    for user in users:
        user_plays = random.choices(plays, weights=[count for _, count in plays], k=play_records_per_user)
        play_records.extend([(user, song_id) for song_id, _ in user_plays])

    return play_records


def write_play_records_to_csv(play_records, output_file):
    """
    Write the generated play records to a CSV file.

    Args:
        play_records (list[tuple]): List of play records, where each record is a tuple containing (user_id, song_id).
        output_file (file-like object): Open file object to which the play records will be written.
    """
    writer = csv.writer(output_file)
    writer.writerow(["User ID", "Song ID"])
    for user_id, song_id in play_records:
        writer.writerow([user_id, song_id])
@app.route('/generate_play_records', methods=['POST'])
def generate_play_records_api():
    data = request.get_json()
    audio_files_directory = data.get('audio_files_directory')
    users_csv_file = data.get('users_csv_file')
    output_csv_file = data.get('output_csv_file')
    avg_plays_per_song = data.get('avg_plays_per_song')
    play_records_per_user = data.get('play_records_per_user')

    if not audio_files_directory or not users_csv_file or not output_csv_file:
        return jsonify(error="Missing one or more required parameters: 'audio_files_directory', 'users_csv_file', 'output_csv_file'"), 400

    try:
        process_audio_files(audio_files_directory)
        users = read_users_from_csv(users_csv_file)
        songs = extract_song_custom_ids(audio_files_directory)

        play_records = generate_play_records(users, songs, avg_plays_per_song, play_records_per_user)
        with open(output_csv_file, 'w', newline='') as output_file:
            write_play_records_to_csv(play_records, output_file)

        return jsonify(success=True, message="Play records generated and saved to CSV")
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)