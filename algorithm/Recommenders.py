import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

import collaborative_filtering as coll

app = Flask(__name__)

# 加载数据
song_count_df = pd.read_csv('../data/song_playcount_df.csv')
play_count_subset = pd.read_csv('../data/user_playcount_df.csv')
triplet_dataset_sub_song_merged = pd.read_csv('../data/complete_dataset.csv')

song_count_subset = song_count_df.head(n=500)
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song_id.isin(song_subset)]

train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size=0.30, random_state=0)

# 初始化模型
is_model = coll.item_similarity_recommender_py()
is_model.create(train_data, 'user', 'song')


@app.route('/recommendations/<string:user_id>', methods=['GET'])
def get_recommendations(user_id):
    try:
        # 执行推荐
        df_recs = is_model.recommend(user_id)
        recommended_items = df_recs.to_dict('records')  # 转换为字典列表，便于序列化为JSON
        return jsonify(recommended_items)
    except Exception as e:
        return jsonify(error=str(e)), 500  # 返回HTTP状态码500（服务器内部错误）及错误信息

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)