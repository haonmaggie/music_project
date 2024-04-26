import os
import random
import csv
import uuid
from typing import Union

import numpy as np
import itertools
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
    # 遍历所有TXXX帧，寻找那些文本信息不是空的且描述为空的帧
    frames_to_update = [frame for frame in audio.getall('TXXX')
                        if frame.desc is None or frame.desc == ''
                        and frame.text[0] != '']

    # 更新这些帧的描述
    for frame in frames_to_update:
        frame.desc = 'Custom ID'

    # 保存更改
    audio.save(v2_version=3)  # 保存为ID3v2.3版本
    #print("ID3标签：",audio)

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
def read_values_from_csv(csv_file_path):
    values_list = []
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        # 跳过首行（列名）
        next(reader)

        for row in reader:
            values_list.extend(row)

    return values_list


import csv

def generate_and_write_play_records_to_csv(
    users,
    songs,
    min_plays,
    max_plays,
    num_random_songs_per_user,
    num_random_users,
    output_file_path
):
    """
    生成用户-歌曲-播放次数记录列表，并将记录写入CSV文件。

    该函数接受一个用户ID列表和一个歌曲自定义ID列表作为输入，为随机选择的一些用户随机生成指定数量的歌曲的播放次数记录。每条记录包含用户ID、歌曲ID以及对应的播放次数。生成的记录将被写入指定的CSV文件。

    参数:
        users (List[str]): 用户ID列表，包含待生成播放记录的用户标识。
        songs (List[str]): 歌曲自定义ID列表，代表可供用户播放的歌曲集合。
        min_plays (int): 播放次数的最小值，默认为5。每条记录中的播放次数将在此范围内随机生成。
        max_plays (int): 播放次数的最大值，默认为50。每条记录中的播放次数将在此范围内随机生成。
        num_random_songs_per_user (int): 每个用户随机选择的歌曲数量。
        num_random_users (int): 随机选择的用户数量。
        output_file_path (str): 目标CSV文件路径，默认为 "play_records.csv"。

    返回:
        None: 不返回任何值，直接将生成的播放记录写入指定的CSV文件。
    """
    def generate_plays_for_user(user_id: str):
        """
        为单个用户生成播放记录。

        该辅助函数接收一个用户ID，针对该用户与给定的歌曲列表，为随机选择的一些歌曲生成一条包含随机播放次数的记录。

        参数:
            user_id (str): 需要为其生成播放记录的用户ID。

        返回:
            List[Dict[str, Union[str, int]]]: 该用户对应的播放记录列表，每条记录的结构与主函数返回的记录相同。
        """

        random_song_subset = random.sample(songs, num_random_songs_per_user)
        user_play_records = []
        for song in random_song_subset:
            play_count = random.randint(min_plays, max_plays)
            user_play_records.append({
                "user": user_id,
                "song_id": song,
                "play_count": play_count
            })
        return user_play_records

    # 随机选择用户并合并他们的播放记录
    random_user_subset = random.sample(users, num_random_users)
    all_play_records = []
    for user in random_user_subset:
        user_records = generate_plays_for_user(user)
        all_play_records.extend(user_records)

    # Write play records to CSV file
    with open(output_file_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "song_id", "play_count"])
        for record in all_play_records:
            writer.writerow([record["user"], record["song_id"], record["play_count"]])

@app.route('/generate_play_records', methods=['POST'])
def generate_play_records_api():
    data = request.get_json()
    audio_files_directory = data.get('audio_files_directory')
    users_csv_file = data.get('users_csv_file')
    output_csv_file = data.get('output_csv_file')
    min_plays = data.get('min_plays')
    max_plays = data.get('max_plays')
    num_random_songs_per_user = data.get('num_random_songs_per_user')
    num_random_users = data.get('num_random_users')

    if not audio_files_directory or not users_csv_file or not output_csv_file:
        return jsonify(error="Missing one or more required parameters: 'audio_files_directory', 'users_csv_file', 'output_csv_file'"), 400

    try:
        process_audio_files(audio_files_directory)
        users = read_values_from_csv(users_csv_file)
        songs = extract_song_custom_ids(audio_files_directory)
        generate_and_write_play_records_to_csv(users, songs, min_plays, max_plays, num_random_songs_per_user, num_random_users, output_csv_file)
        return jsonify(success=True, message="Play records generated and saved to CSV")
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True)