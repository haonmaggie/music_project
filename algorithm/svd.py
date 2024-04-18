import numpy as np
import pandas as pd
import recommenders as Recommenders

# 加载数据集
song_count_df = pd.read_csv('../data/song_playcount_df.csv')
play_count_subset = pd.read_csv('../data/user_playcount_df.csv')
triplet_dataset_sub_song_merged = pd.read_csv('../data/complete_dataset.csv')

song_count_subset = song_count_df.head(n=500)
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song.isin(song_subset)]

triplet_dataset_sub_song_merged_sum_df = triplet_dataset_sub_song_merged[['user','play_count']].groupby('user').sum().reset_index()
triplet_dataset_sub_song_merged_sum_df.rename(columns={'play_count':'total_play_count'},inplace=True)
triplet_dataset_sub_song_merged = pd.merge(triplet_dataset_sub_song_merged,triplet_dataset_sub_song_merged_sum_df)

# print(triplet_dataset_sub_song_merged.head())

triplet_dataset_sub_song_merged['fractional_play_count'] = triplet_dataset_sub_song_merged['play_count']/triplet_dataset_sub_song_merged['total_play_count']
triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='0eWtfZi67r6GktMV'][['user','song_id','play_count','fractional_play_count']].head()
# print(triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.user =='0eWtfZi67r6GktMV'][['user','song_id','play_count','fractional_play_count']].head())

# 根据用户进行分组，计算每个用户的总的播放总量，然后用每首歌的播放总量相除，得到每首歌的分值
# 最后一列特征fractional_play_count就是用户对每首歌曲的评分值
from scipy.sparse import coo_matrix

small_set = triplet_dataset_sub_song_merged
user_codes = small_set.user.drop_duplicates().reset_index()
song_codes = small_set.song.drop_duplicates().reset_index()
user_codes.rename(columns={'index':'user_index'}, inplace=True)
song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)
user_codes['us_index_value'] = list(user_codes.index)
small_set = pd.merge(small_set,song_codes,how='left')
small_set = pd.merge(small_set,user_codes,how='left')
mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]
data_array = mat_candidate.fractional_play_count.values
row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values

data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)

# 设置随机抽取的用户数量
num_users_to_sample = 3  # 示例：抽取5个用户

# 从row_array（用户索引）中随机抽取指定数量的用户
random_user_indices = np.random.choice(row_array, size=num_users_to_sample, replace=False)

# 将随机用户索引数组转换为列表
random_user_indices_list = random_user_indices.tolist()

# print("随机抽取的用户索引列表：", random_user_indices_list)

import math as mt
from scipy.sparse.linalg import *  # used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix


def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(np.diag(s), dtype=np.float32)  # 直接使用对角矩阵S
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S * Vt
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float32)
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
K = 50
urm = data_sparse
MAX_PID = urm.shape[1]
MAX_UID = urm.shape[0]

U, S, Vt = compute_svd(urm, K)

uTest = random_user_indices_list

uTest_recommended_scores = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)

# 检查urm与small_set的歌曲索引一致性
assert set(urm.col).issubset(set(small_set['so_index_value'].values)), "urm与small_set的歌曲索引不一致"

for user_idx, user in enumerate(uTest):
    print("当前待推荐用户编号 {}".format(user))
    rank_value = 1
    min_score = np.min(uTest_recommended_scores[user_idx, :])
    max_score = np.max(uTest_recommended_scores[user_idx, :])
    for score_idx, score in enumerate(uTest_recommended_scores[user_idx, :10]):  # 修改：遍历推荐度分数的索引和值
        normalized_score = (score - min_score) / (max_score - min_score) * 100
        song_index = uTest_recommended_scores[user_idx, score_idx]  # 获取当前推荐歌曲的索引
        song_details = small_set[small_set.so_index_value == song_index].drop_duplicates('so_index_value')[
            ['song', 'artist']]

        if len(song_details['song']) > 0 and len(song_details['artist']) > 0:  # 新增：检查列表长度
            print("推荐编号： {} 推荐歌曲： {} 作者： {}（推荐度： {:.2f}）".format(rank_value, list(song_details['song'])[0],
                                                                              list(song_details['artist'])[0], normalized_score))
            rank_value += 1
        else:
            print("推荐编号： {} 未能找到对应歌曲和作者信息（推荐度： {:.2f}）".format(rank_value, normalized_score))
            rank_value += 1

