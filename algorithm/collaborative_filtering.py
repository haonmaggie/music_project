import numpy as np
import pandas as pd

class popularity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.popularity_recommendations = None

    # 创建基于流行度的推荐系统模型
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

        # 计算每首唯一歌曲的用户ID计数作为推荐得分
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns={user_id: 'score'}, inplace=True)

        # 根据推荐得分对歌曲进行排序
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending=[0, 1])

        # 依据得分生成推荐排名
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

        # 获取前10个推荐
        self.popularity_recommendations = train_data_sort.head(10)

    # 使用基于流行度的推荐系统模型
    # 为用户生成推荐
    def recommend(self, user_id):
        user_recommendations = self.popularity_recommendations

        # 添加用于生成推荐的用户ID列
        user_recommendations['user_id'] = user_id

        # 将用户ID列移动至最前面
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations


# Class for Item similarity based Recommender System model
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    # 获取给定用户对应的唯一项目（歌曲）
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items

    # 获取给定项目（歌曲）对应的唯一用户
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    # 获取训练数据中所有的唯一项目（歌曲）
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    # 构建共现矩阵
    def construct_cooccurence_matrix(self, user_songs, all_songs):

        ####################################
        # 获取用户_songs中所有歌曲的所有用户。
        # 现在要计算的是给我选中的测试用户推荐什么
        # 流程如下
        # 1. 先把选中的测试用户听过的歌曲都拿到
        # 2. 找出这些歌曲中每一个歌曲都被那些其他用户听过
        # 3. 在整个歌曲集中遍历每一个歌曲，计算它与选中测试用户中每一个听过歌曲的Jaccard相似系数
        # 通过听歌的人的交集与并集情况来计算
        ####################################
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        ###############################################
        # 初始化大小为 len(user_songs) × len(songs) 的项目共现矩阵
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        #############################################################
        # 计算用户歌曲与训练数据中所有唯一歌曲之间的相似度
        #############################################################
        for i in range(0, len(all_songs)):
            # 计算项目i（歌曲）的唯一听众（用户）
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())

            for j in range(0, len(user_songs)):

                # 获取项目j（歌曲）的唯一听众（用户）
                users_j = user_songs_users[j]

                # 计算歌曲i和j听众的交集
                users_intersection = users_i.intersection(users_j)

                # 计算共现矩阵[i,j]作为Jaccard指数
                if len(users_intersection) != 0:
                    # 计算歌曲i和j听众的并集
                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    # Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("共现矩阵中非零值的数量：%d" % np.count_nonzero(cooccurence_matrix))

        # 计算共现矩阵中所有用户歌曲的加权平均得分
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # 根据得分值对user_sim_scores的索引进行排序，同时保留相应得分
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # 以以下内容创建一个DataFrame
        columns = ['user_id', 'song_id', 'score', 'rank']
        # index = np.arange(1) # 数组，用于存储样本数量
        df = pd.DataFrame(columns=columns)

        # 用基于项目的前10个推荐填充DataFrame
        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)] = [user, all_songs[sort_index[i][1]], sort_index[i][0], rank]
                rank = rank + 1

        # 处理无推荐的情况
        if df.shape[0] == 0:
            print("当前用户没有可用于训练基于项目相似度推荐模型的歌曲。")
            return -1
        else:
            # 区间0-1
            # 计算 score 列的最小值和最大值
            min_score = df['score'].min()
            max_score = df['score'].max()

            # 对 score 列进行线性归一化处理
            df['normalized_score'] = (df['score'] - min_score) / (max_score - min_score)

            # 确保归一化后的得分在0到1之间
            df['scaled_score'] = np.clip(df['normalized_score'], 0, 1)

        return df

    # 创建基于项目相似度的推荐系统模型
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    # 使用基于物品相似度的推荐系统模型生成推荐
    def recommend(self, user):

        ########################################
        # A. 获取该用户的所有唯一歌曲
        ########################################
        user_songs = self.get_user_items(user)

        print("用户拥有的唯一歌曲数量：%d" % len(user_songs))

        ######################################################
        # B. 获取训练数据中所有的唯一项目（歌曲）
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("训练集中唯一歌曲数量：%d" % len(all_songs))

        ###############################################
        # C. 构建大小为 len(user_songs) × len(songs) 的项目共现矩阵
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. 利用共现矩阵生成推荐
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return df_recommendations

    # Get similar items to given items
    def get_similar_items(self, item_list):
        user_songs = item_list

        ######################################################
        # B. 获取训练数据中所有的唯一项目（歌曲）
        ######################################################
        all_songs = self.get_all_items_train_data()

        print("训练集中唯一歌曲数量：%d" % len(all_songs))

        ###############################################
        # C. 构建大小为 len(user_songs) × len(songs) 的项目共现矩阵
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        #######################################################
        # D. 利用共现矩阵生成推荐
        #######################################################
        user = ""  # 空字符串作为用户标识
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)

        return df_recommendations





