import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

complete_dataset = pd.read_csv('../data/complete_dataset.csv')

# 按歌曲名字来统计其播放量的总数
popular_songs = complete_dataset[['song_name', 'play_count']].groupby('song_name').sum().reset_index()
user_song_count_distribution = complete_dataset[['user','song_name']].groupby('user').count().reset_index().sort_values(
by='song_name',ascending = False)
user_song_count_distribution.song_name.describe()

# 对结果进行排序
popular_songs_top_20 = popular_songs.sort_values('play_count', ascending=False).head(n=20)

# 设置中文字体支持
plt.rcParams['font.family'] = ['SimHei']  # 替换为系统中已安装的其他支持中文的字体名

# 转换成list格式方便画图
objects = list(popular_songs_top_20['song_name'])

# 设置位置
y_pos = np.arange(len(objects))

# 对应结果值
performance = list(popular_songs_top_20['play_count'])

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))  # 可以适当调整图形大小以适应更多歌曲标签
ax.bar(y_pos, performance, align='center', alpha=0.5)

# 调整字体大小、垂直旋转角度、水平对齐方式及防止溢出
ax.set_xticks(y_pos)
ax.set_xticklabels(objects, rotation='vertical', fontsize='small', ha='right')
ax.tick_params(axis='x', pad=10)  # 增加标签与条形间的距离，避免文字重叠

ax.set_ylabel('播放次数')
ax.set_title('最受欢迎歌曲排行榜')




fig.tight_layout()  # 自动调整布局以避免标签被裁剪
plt.show()