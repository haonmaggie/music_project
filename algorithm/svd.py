import numpy as np
import pandas as pd
import io
import  sys
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds
from flask import Flask, jsonify, request

app = Flask(__name__)

def load_data(song_playcount_df_file,user_playcount_df_file,complete_dataset_file):
    # 加载数据集
    song_count_df = pd.read_csv(io.BytesIO(song_playcount_df_file.read()))
    play_count_subset = pd.read_csv(io.BytesIO(user_playcount_df_file.read()))
    triplet_dataset_sub_song_merged = pd.read_csv(io.BytesIO(complete_dataset_file.read()))

    song_count_subset = song_count_df.head(n=500)
    #user_subset = list(play_count_subset.user)
    song_subset = list(song_count_subset.song_id)
    #triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]

    triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user_id','play_count']].groupby('user_id').sum().reset_index()
    triplet_dataset_sub_song_merged_sum_df.rename(columns={'play_count':'total_play_count'},inplace=True)
    triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)

    # print(triplet_dataset_sub_song_merged.head())

    triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['play_count']/triplet_dataset_sub_song_merged['total_play_count']
    triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user_id =='0eWtfZi67r6GktMV'][['user_id', 'song_id', 'play_count', 'fractional_play_count']].head()
    # print(triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='0eWtfZi67r6GktMV'][['user','song_id','play_count','fractional_play_count']].head())
    return triplet_dataset_sub_song_merged


# 根据用户进行分组，计算每个用户的总的播放总量，然后用每首歌的播放总量相除，得到每首歌的分值
# 最后一列特征fractional_play_count就是用户对每首歌曲的评分值
from scipy.sparse import coo_matrix


def generate_user_index_mapping(user_ids):
    index_mapping = {user_id: index for index, user_id in enumerate(user_ids)}
    return index_mapping


def process_data(triplet_dataset_sub_song_merged, user_ids):
    print("列表：", user_ids)

    # 确保user_ids中的用户在数据集中存在
    valid_user_ids = triplet_dataset_sub_song_merged['user_id'].isin(user_ids)
    small_set = triplet_dataset_sub_song_merged[valid_user_ids]

    if small_set.empty:
        raise ValueError("No data found for the provided user_ids.")

    # 生成用户和歌曲的索引映射
    user_index_mapping = generate_user_index_mapping(small_set['user_id'].unique())
    song_index_mapping = generate_user_index_mapping(small_set['song_id'].unique())

    # 将索引映射添加到数据集中
    small_set['user_index'] = small_set['user_id'].map(user_index_mapping)
    small_set['song_index'] = small_set['song_id'].map(song_index_mapping)

    # 构建稀疏矩阵的准备步骤
    mat_candidate = small_set[['user_index', 'song_index', 'fractional_play_count']]
    data_array = mat_candidate.fractional_play_count.values
    row_array = mat_candidate.user_index.values
    col_array = mat_candidate.song_index.values

    data_sparse = coo_matrix((data_array, (row_array, col_array)), dtype=float)
    # 获取已处理数据集中的唯一用户索引列表，这应该与我们生成的映射相对应
    unique_user_indices = small_set['user_index'].unique()

    # 返回处理后的数据集、稀疏矩阵以及用户索引列表
    return small_set, data_sparse, list(unique_user_indices)
# print("随机抽取的用户索引列表：", random_user_indices_list)

from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(np.diag(s), dtype=np.float32)  # 直接使用对角矩阵S
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    max_uid = urm.shape[0]
    max_pid = urm.shape[1]
    rightTerm = S * Vt
    max_recommendation = 85
    estimatedRatings = np.zeros(shape=(max_uid, max_pid), dtype=np.float32)
    recomendRatings_scores = np.zeros(shape=(len(uTest), max_recommendation), dtype=np.float32)  # 新增：存储推荐歌曲的评分值

    for userTest_idx, userTest in enumerate(uTest):
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = prod.todense()

        # 新增：按评分值降序排序，并保存前max_recommendation个评分值
        top_k_indices = estimatedRatings[userTest, :].argsort()[::-1][:max_recommendation]  # 对单个用户评分向量进行降序排列
        recomendRatings_scores[userTest_idx, :] = top_k_indices

    return recomendRatings_scores

# 额外指定一个指标K值 即选择前多少个特征值来做近似代表，也就是S矩阵中的数量
# 如果K值较大整体的计算效率会慢一些但是会更接近真实结果
# PID表示最开始选择的部分歌曲，UID表示选择的部分用户

def generate_recommendations(user_ids, k, triplet_dataset_sub_song_merged):
    # 直接使用用户ID列表，无需从process_data中返回的user_data获取，因为我们已经确保了user_ids的有效性
    small_set, data_sparse, user_id_list = process_data(triplet_dataset_sub_song_merged, user_ids)
    urm = data_sparse
    U, S, Vt = compute_svd(urm, k)
    # 直接使用user_ids作为uTest
    uTest_recommended_scores = compute_estimated_matrix(urm, U, S, Vt, user_id_list, k, True)

    # 确保urm与small_set的歌曲索引一致性检查保持不变
    assert set(urm.col).issubset(set(small_set['song_index'].values)), "urm与small_set的歌曲索引不一致"

    return uTest_recommended_scores, user_id_list, small_set

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # 获取文本参数
    output_file = request.form.get('output_file')  # 对于 'output_file'
    user_id_str = request.form.get('user_id')
    k_str = request.form.get('k_str')
    # user_id以逗号分隔，去除前后可能存在的空白字符后分割成列表
    user_ids = [uid.strip() for uid in user_id_str.split(',') if uid.strip()]
    # 获取文件
    song_playcount_df_file = request.files.get('song_playcount_df')
    user_playcount_df_file = request.files.get('user_playcount_df')
    complete_dataset_file = request.files.get('complete_dataset')
    # 加载数据集和数据预处理
    k = int(k_str)
    triplet_dataset_sub_song_merged = load_data(song_playcount_df_file, user_playcount_df_file, complete_dataset_file)
    recommendations, user_data, small_set = generate_recommendations(user_ids, k, triplet_dataset_sub_song_merged)

    recommendation_list = []
    for user_idx, user in enumerate(user_data):
        min_score = np.min(recommendations[user_idx, :])
        max_score = np.max(recommendations[user_idx, :])
        for score_idx, score in enumerate(recommendations[user_idx, :10]):  # 修改：遍历推荐度分数的索引和值
            # 直接归一化到0-1区间
            normalized_score = (score - min_score) / (max_score - min_score)
            song_index = recommendations[user_idx, score_idx]  # 获取当前推荐歌曲的索引
            song_details = small_set[small_set.song_index == song_index].drop_duplicates('song_index')[
                ['song', 'artist']]

            if not song_details.empty:
                song_name = song_details['song'].iloc[0]
                artist = song_details['artist'].iloc[0]
                recommendation_list.append({
                    "用户编号": user_idx,
                    "推荐编号": score_idx + 1,  # 推荐编号从1开始计数
                    "歌曲索引": song_index,
                    "歌曲名": song_name,
                    "作者": artist,
                    "推荐度(0-1)": round(normalized_score, 4)  # 保留四位小数
                })

        # 将列表转换为DataFrame
        df_recommendations = pd.DataFrame(recommendation_list)
        df_recommendations.to_excel(output_file, index=False)

    return jsonify({'message': f'Recommendations written to {output_file}'})

if __name__ == '__main__':
    app.run( debug=True)

