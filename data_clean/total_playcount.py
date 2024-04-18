import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data/play_records.csv')

# 提取用户和播放次数
df['play_count'] = df['play_count'].astype(int)  # 假设'Play_Count'列在CSV中为整数类型
# 提取用户和播放次数
df['play_count'] = df['play_count'].astype(int)  # 假设'Play_Count'列在CSV中为整数类型

# 统计歌曲播放次数,统计用户播放次数
song_output_df = df.groupby('song')['play_count'].sum().reset_index()
user_output_df = df.groupby('user')['play_count'].sum().reset_index()

# 前400名用户的播放量占总体的比例
total_play_count = sum(song_output_df .play_count) # 所有歌曲的播放量
print((float(user_output_df.head(n=400).play_count.sum()) / total_play_count) * 100) # 前400用户播放总量占比
user_count_subset = user_output_df.head(n=400)
# 前300首歌曲的播放量占总体的比例
print((float(song_output_df.head(n=300).play_count.sum())/total_play_count)*100) # 前300歌曲播放总量占比
song_count_subset = song_output_df.head(n=300)


user_subset = list(user_count_subset.user)
song_subset = list(song_count_subset.song)