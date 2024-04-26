from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

@app.route('/generate_playcounts_data', methods=['POST'])
def generate_playcounts_data():
    data = request.get_json()
    input_csv_file = data.get('play_records_file')
    song_output_csv_file = data.get('song_playcounts_file')
    user_output_csv_file = data.get('user_playcounts_file')

    # 读取CSV文件
    df = pd.read_csv(input_csv_file)

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)

    # 统计歌曲播放次数
    song_output_df = df.groupby('song_id')['play_count'].sum().reset_index()
    sorted_song_df = song_output_df.sort_values(by='play_count', ascending=False)

    # 写入歌曲播放次数到指定CSV文件
    sorted_song_df.to_csv(song_output_csv_file, index=False)

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)
    output_df = df.groupby('user_id')['play_count'].sum().reset_index()

    # 将用户按照播放量从高到低排序
    sorted_df = output_df.sort_values(by='play_count', ascending=False)

    # 写入用户播放次数到指定CSV文件
    sorted_df.to_csv(user_output_csv_file, index=False)

    # 返回成功消息
    return jsonify({'message': 'Playcounts data generated successfully.'})

if __name__ == '__main__':
    app.run(debug=True)