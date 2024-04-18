import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
complete_dataset = pd.read_csv('../data/complete_dataset.csv')

# 统计最受欢迎的专辑
popular_albums = complete_dataset[['album_name', 'play_count']].groupby('album_name').sum().reset_index()

popular_albums_top_20 = popular_albums.sort_values('play_count', ascending=False).head(n=20)

# 设置中文字体支持
plt.rcParams['font.family'] = ['SimHei']  # 替换为系统中已安装的其他支持中文的字体名

# 转换成list格式方便画图
objects = list(popular_albums_top_20['album_name'])
performance = list(popular_albums_top_20['play_count'])  # 修改这里，使用播放次数列表

# 设置位置
y_pos = np.arange(len(objects))

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))  # 可以适当调整图形大小以适应更多歌曲标签
ax1.bar(y_pos, performance, align='center', alpha=0.5)

# 调整字体大小、垂直旋转角度、水平对齐方式及防止溢出
ax1.set_xticks(y_pos)
ax1.set_xticklabels(objects, rotation='vertical', fontsize='small', ha='right')
ax1.tick_params(axis='x', pad=10)  # 增加标签与条形间的距离，避免文字重叠

ax1.set_ylabel('播放次数')
ax1.set_title('最受欢迎专辑排行榜')

fig.tight_layout()
plt.show()