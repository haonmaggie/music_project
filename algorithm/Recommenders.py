import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import collaborative_filtering as coll

song_count_df = pd.read_csv('../data/song_playcount_df.csv')
play_count_subset = pd.read_csv('../data/user_playcount_df.csv')
triplet_dataset_sub_song_merged = pd.read_csv('../data/complete_dataset.csv')

song_count_subset = song_count_df.head(n=500)
user_subset = list(play_count_subset.user)
song_subset = list(song_count_subset.song)
triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[triplet_dataset_sub_song_merged.song_id.isin(song_subset)]

# print("数据集前几条记录:", triplet_dataset_sub_song_merged_sub[:5])

train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size = 0.30, random_state=0)
is_model = coll.item_similarity_recommender_py()
is_model.create(train_data, 'user', 'song')
user_id = list(train_data.user)[7]
user_items = is_model.get_user_items(user_id)

#执行推荐
user_id = '8sp3qlAHqWADjWJE'
df_recs= is_model.recommend(user_id)
print(df_recs)

