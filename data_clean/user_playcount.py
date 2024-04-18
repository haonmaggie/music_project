from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/generate_playcounts_data', methods=['GET'])
def generate_playcounts_data():
    # 读取CSV文件
    df = pd.read_csv('../data/play_records.csv')

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)

    # 统计歌曲播放次数
    song_output_df = df.groupby('song')['play_count'].sum().reset_index()
    sorted_song_df = song_output_df.sort_values(by='play_count', ascending=False)

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)
    output_df = df.groupby('user')['play_count'].sum().reset_index()

    # 将用户按照播放量从高到低排序
    sorted_df = output_df.sort_values(by='play_count', ascending=False)

    # 将结果序列化为JSON并返回
    return jsonify({
        'song_playcounts': sorted_song_df.to_dict('records'),
        'user_playcounts': sorted_df.to_dict('records')
    })

if __name__ == '__main__':
    app.run(debug=True)