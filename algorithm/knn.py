import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# 假设已预处理并加载数据集为 DataFrame，且已处理好非数值特征
data = pd.read_csv('../data/complete_dataset.csv')

# 选择相关特征
features = ['artist', 'year']  # 示例，实际应根据数据集调整
target_users = 'user'
target_songs = 'song_id'

# 构建用户-物品矩阵
user_song_matrix = pd.pivot_table(
    data,
    index=target_users,
    columns=target_songs,
    values='play_count',
    fill_value=0
).fillna(0)
user_song_matrix = csr_matrix(user_song_matrix)

# 特征缩放（此处仅针对数值特征，非数值特征需提前编码）
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(user_song_matrix.todense())

# 训练KNN模型
k = 10  # 示例K值，可根据实际情况调整
nn = NearestNeighbors(metric='cosine', n_neighbors=k)
nn.fit(scaled_matrix)

# 实现推荐
def recommend_music(user_id, top_n=10):
    # 查找用户ID对应的行索引
    user_row = data[data[target_users] == user_id].index[0]

    user_vector = scaled_matrix[user_row].reshape(1, -1)
    distances, indices = nn.kneighbors(user_vector)

    # 对邻居的播放记录进行加权求和
    neighbor_play_counts = user_song_matrix[indices[0]].toarray()  # 取出邻居的播放记录
    weights = 1 / distances[0]  # 以距离倒数作为权重
    weighted_play_counts = neighbor_play_counts * weights[:, np.newaxis]

    # 汇总并排序
    summed_play_counts = np.sum(weighted_play_counts, axis=0)
    min_score = summed_play_counts.min()
    max_score = summed_play_counts.max()

    # 线性映射到 [1, 100] 区间
    normalized_scores = 1 + ((summed_play_counts - min_score) / (max_score - min_score)) * (100 - 1)

    sorted_song_indices = normalized_scores.argsort()[::-1][:top_n]

    # 映射索引到原始歌曲ID字符串
    recommended_song_ids = data[target_songs].iloc[sorted_song_indices].tolist()
    recommended_song_scores = normalized_scores[sorted_song_indices].tolist()
    # 在返回推荐歌曲ID和分数前，对推荐度进行四舍五入
    recommended_song_scores = np.around(recommended_song_scores, decimals=4).tolist()

    return recommended_song_ids, recommended_song_scores

# 示例：为用户 "0eWtfZi67r6GktMV" 推荐10首歌曲
recommended_song_ids, recommended_song_scores = recommend_music("0eWtfZi67r6GktMV", top_n=10)
print(f"Recommended songs for User '0eWtfZi67r6GktMV':")

for song_id, score in zip(recommended_song_ids, recommended_song_scores):
    print(f"{song_id}: {score}")