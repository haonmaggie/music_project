import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 使用“黑体”作为默认字体（请替换为实际已安装且支持所需字符的中文字体）
# 读取完整数据集
complete_dataset = pd.read_csv('../data/complete_dataset.csv')

# 统计用户播放量分布
user_play_count_distribution = complete_dataset[['user', 'song_name']].groupby('user').count().reset_index().sort_values(
    by='song_name', ascending=False)
user_play_count_summary = user_play_count_distribution['song_name'].describe()

# 播放次数数据
x = user_play_count_distribution['song_name']

# 绘制直方图
n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.75)

# 图表元素设置（中文）
plt.xlabel('播放次数')
plt.ylabel('用户数')
plt.title(r'用户播放次数分布直方图')
plt.grid(True)

plt.show()