import os
import random
import csv
import uuid
import numpy as np
from collections import Counter
from mutagen.id3 import ID3, TXXX
from flask import Flask, jsonify, request

app = Flask(__name__)

# 读取用户ID列表
with open("../data/users.txt", "r") as user_file:
    users = user_file.read().splitlines()

# 指定MP3文件所在的目录路径
mp3_directory = "E:/music/华语歌曲"
# 假设平均每个用户对每首歌的播放次数为 AVG_PLAYS_PER_SONG
AVG_PLAYS_PER_SONG = 2.5
# 扫描目录获取MP3文件及对应的自定义ID（歌曲编号）
songs = []
print("Scanning MP3 files in directory:", mp3_directory)


# 遍历目录及其子目录下的所有MP3文件
def process_directory(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.mp3'):
                filepath = os.path.join(dirpath, filename)

                # 生成类似UUID格式的编号
                unique_id = str(uuid.uuid4()).upper().replace('-', '')

                # 打开MP3文件的ID3标签
                audio = ID3(filepath)

                # 创建一个TXXX帧用于存储自定义编号
                custom_id_frame = TXXX(encoding=3, description='Custom ID', text=unique_id)

                # 添加或更新ID3标签中的自定义编号
                if 'TXXX:Custom ID' not in audio:
                    audio.add(custom_id_frame)
                else:
                    audio['TXXX:Custom ID'] = custom_id_frame

                # 保存更改
                audio.save(v2_version=3)  # 保存为ID3v2.3版本


for root, dirs, files in os.walk(mp3_directory):
    for filename in files:
        if filename.endswith(".mp3"):
            try:
                audio = ID3(os.path.join(root, filename))
                custom_id = None
                for frame in audio.values():
                    if isinstance(frame, TXXX) and frame.desc == 'Custom ID':
                        custom_id = frame.text[0]
                        break

                if custom_id is not None:
                    songs.append(custom_id)
                    print(f"Found custom ID '{custom_id}' in file '{os.path.join(root, filename)}'")
                else:
                    print(f"No custom ID found in file '{os.path.join(root, filename)}'")
            except Exception as e:
                print(f"Error reading ID3 tag from file {os.path.join(root, filename)}: {e}")

# 定义每个用户生成的播放记录条数
PLAY_RECORDS_PER_USER = 50

# 模拟用户播放记录并统计播放次数
def generate_play_records(users, songs, avg_plays_per_song, play_records_per_user):
    play_records = []

    for user in users:
        for _ in range(play_records_per_user):
            if not songs or len(songs) == 0:
                raise ValueError("The 'songs' list is empty. Unable to randomly choose a song.")
            song = random.choice(songs)
            # 使用泊松分布模拟每首歌被该用户播放的次数
            play_count = np.random.poisson(avg_plays_per_song)
            play_record = {"user": user, "song": song, "play_count": play_count}
            play_records.append(play_record)

    return play_records

# 将播放次数转换为CSV格式并写入文件
def write_play_records_to_csv(play_records, output_file):
    writer = csv.writer(output_file)
    writer.writerow(["user", "song", "play_count"])  # 更改列名以匹配play_records结构

    for record in play_records:
        writer.writerow([record["user"], record["song"], record["play_count"]])

    output_file.close()  # 添加这一行以确保文件被正确关闭

@app.route('/generate_play_records', methods=['POST'])
def generate_play_records_api():
    data = request.get_json()
    users = data.get('users')
    avg_plays_per_song = data.get('avg_plays_per_song', AVG_PLAYS_PER_SONG)
    play_records_per_user = data.get('play_records_per_user', PLAY_RECORDS_PER_USER)

    if not users:
        return jsonify(error="Missing 'users' parameter"), 400

    try:
        play_records = generate_play_records(users, songs, avg_plays_per_song, play_records_per_user)
        write_play_records_to_csv(play_records, '../data/play_records.csv')

        return jsonify(success=True, message="Play records generated and saved to CSV")
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)