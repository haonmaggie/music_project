import pandas as pd
import os
import csv
from mutagen.id3 import ID3
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/generate_complete_dataset', methods=['POST'])
def generate_complete_dataset():
    # 接收请求体中的CSV文件路径
    mp3_csv_path = request.json.get('mp3_csv_path')
    play_records_csv_path = request.json.get('play_records_csv_path')
    output_csv_path = request.json.get('output_csv_path')

    if mp3_csv_path is None or play_records_csv_path is None:
        return jsonify({"status": "error", "message": "Missing required parameter(s): 'mp3_csv_path' and/or 'play_records_csv_path'."}), 400

    try:
        # 加载数据
        mp3_df = pd.read_csv(mp3_csv_path)
        play_records_df = pd.read_csv(play_records_csv_path)

        # 数据清洗与重命名列
        play_records_df.rename(columns={
            'user_id': 'user_id',
            'song_id': 'song_id',
            'play_count': 'play_count'
        }, inplace=True)

        mp3_df.rename(columns={
            'custom_id': 'song_id',
            'title': 'song',
            'album': 'album'
        }, inplace=True)

        # 合并数据集
        merged_df = pd.merge(mp3_df, play_records_df[['user_id', 'song_id', 'play_count']], on='song_id', how='left')

        # 筛选所需的列
        desired_columns = ['user_id', 'song_id', 'play_count', 'song', 'album', 'artist', 'year']
        merged_df = merged_df[desired_columns]

        # 保存到指定路径
        merged_df.to_csv(output_csv_path, index=False)

        return jsonify({"status": "success", "message": f"Complete dataset generated and saved to {output_csv_path}."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred while processing the data: {str(e)}"}), 500

if __name__ == '__main__':
    app.run( debug=True)