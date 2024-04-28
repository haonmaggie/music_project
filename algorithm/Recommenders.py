import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

import collaborative_filtering as coll

app = Flask(__name__)

@app.route('/process_and_recommend', methods=['POST'])
def process_and_recommend():
    # 获取文本参数
    output_file = request.form.get('output_file')  # 对于 'output_file'
    user_id = request.form.get('user_id')

    # 获取文件
    song_playcount_df_file = request.files.get('song_playcount_df')
    user_playcount_df_file = request.files.get('user_playcount_df')
    complete_dataset_file = request.files.get('complete_dataset')
    muisc_df = request.files.get('muisc_df')
    # 检查是否接收到所有必需的文件
    if not all([song_playcount_df_file, user_playcount_df_file, complete_dataset_file]):
        return jsonify(error="Missing one or more required files"), 400
    # 确保文件有效并转换为 pandas DataFrame
    try:
        music_data_df = pd.read_csv(io.BytesIO(muisc_df.read()))
        song_count_df = pd.read_csv(io.BytesIO(song_playcount_df_file.read()))
        play_count_subset = pd.read_csv(io.BytesIO(user_playcount_df_file.read()))
        triplet_dataset_sub_song_merged = pd.read_csv(io.BytesIO(complete_dataset_file.read()))

        song_count_subset = song_count_df.head(n=500)
        user_subset = list(play_count_subset.user_id)
        song_subset = list(song_count_subset.song_id)
        triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[
            triplet_dataset_sub_song_merged.song_id.isin(song_subset)]
        train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size=0.30, random_state=0)
        # print("数据集前几条记录:", triplet_dataset_sub_song_merged_sub[:5])

        # 初始化模型
        is_model = coll.item_similarity_recommender_py()
        is_model.create(train_data, 'user_id', 'song_id')

        # 执行推荐
        df_recs = is_model.recommend(user_id)
        song_ids = df_recs['song_id']
        song_df = music_data_df[['custom_id', 'title']]  # 正确提取两列的方式

        # 根据song_ids匹配歌名
        filtered_songs = song_df[song_df['custom_id'].isin(song_ids)]
        song_titles = filtered_songs.set_index('custom_id')['title']

        # 在df_recs中添加新列'title'，注意需要保证df_recs的'song_id'与music_data_df的'custom_id'一一对应
        df_recs = df_recs.merge(song_titles, left_on='song_id', right_index=True, how='left')

        # 写入Excel，此时df_recs已包含歌名列
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df_recs.to_excel(writer, sheet_name='Sheet1', index=False)
        return jsonify({'message': f'Recommendations written to {output_file}'})


    except Exception as e:
        return jsonify(error=str(e)), 500  # 返回HTTP状态码500（服务器内部错误）及错误信息



if __name__ == '__main__':
    app.run(debug=True)