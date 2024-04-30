import pandas as pd
from mutagen.id3 import ID3
import secrets
import string
import csv
import numpy as np
import io
import os
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request, render_template, redirect
from flask import current_app

import data_clean.user_song_csv as usc     #生成完整数据集
import data_clean.user_playcount as up     #生成用户-播放、歌曲-播放数据集
import data_clean.song_to_database as std  #音乐分类
import data_clean.user as user             #生成用户id
import data_clean.user_song as us          #生成用户-歌曲-播放数据集
import data_clean.song_to_csv as stc       #生成歌曲转数据集

import algorithm.collaborative_filtering as coll #协同过滤-基于歌曲相似度
import algorithm.svd as svd                     #svd-矩阵分解，模拟评分


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

#第一步 进行音乐分类
@app.route('/test/songToDatabase')
def songtodatabase():
    return render_template("test/songtodatabase.html")
@app.route('/songToDatabase', methods=['POST'])
def organize():
    data = request.get_json()

    input_dir = data.get('input_directory')
    if not input_dir or not os.path.isdir(input_dir):
        return jsonify({'error': 'Invalid input directory'}), 400

    output_dir = data.get('output_directory')
    if not output_dir:
        return jsonify({'error': 'Missing output directory'}), 400

    try:
        std.organize_audio_files(input_dir,output_dir)
        return jsonify({'message': 'Audio files organized successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/test/generate_user_ids')
def userid():
    return render_template("test/userid.html")
# 第二步 随机生成用户数据
def generate_random_string():
    allowed_chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(allowed_chars) for _ in range(16))
@app.route('/generate_user_ids', methods=['POST'])
def generate_user_ids_api():
    try:
        num_users = int(request.json['num_users'])
        out_dir = request.json['out_dir']
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'num_users' parameter. Please provide an integer value."}), 400

    if num_users <= 0 or num_users > 1000:  # 假设最大允许生成1000个用户ID
        return jsonify({"error": f"Number of user IDs requested ({num_users}) is outside the allowed range (1-1000)."}), 400

    users = [generate_random_string() for _ in range(num_users)]

    # 保存用户ID到CSV文件
    user.save_to_csv(users, out_dir)

    return jsonify({"user_ids": users}), 200

# 第三步 生成用户-歌曲-播放数据集
@app.route('/test/play_records')
def playrecords():
    return render_template("test/play_records.html")
@app.route('/generate_play_records', methods=['POST'])
def generate_play_records_api():
    data = request.get_json()
    audio_files_directory = data.get('audio_files_directory')
    users_csv_file = data.get('users_csv_file')
    output_csv_file = data.get('output_csv_file')
    min_plays = data.get('min_plays')
    max_plays = data.get('max_plays')
    num_random_songs_per_user = data.get('num_random_songs_per_user')
    num_random_users = data.get('num_random_users')

    if not audio_files_directory or not users_csv_file or not output_csv_file:
        return jsonify(error="Missing one or more required parameters: 'audio_files_directory', 'users_csv_file', 'output_csv_file'"), 400

    try:
        us.process_audio_files(audio_files_directory)
        users = us.read_values_from_csv(users_csv_file)
        songs = us.extract_song_custom_ids(audio_files_directory)
        min_play = int(min_plays)
        max_play = int(max_plays)
        num_random_user = int(num_random_users)
        num_random_songs_per_us = int(num_random_songs_per_user)
        us.generate_and_write_play_records_to_csv(users, songs, min_play, max_play, num_random_songs_per_us, num_random_user, output_csv_file)
        return jsonify(success=True, message="Play records generated and saved to CSV")
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 500

# 第四步 生成歌曲详情数据集
@app.route('/test/song_detail')
def songdetail():
    return render_template("test/song_detail.html")
@app.route('/extract_audio_info', methods=['POST'])
def extract_audio_info():
    # 接收请求体中的音频文件目录路径和输出CSV文件路径
    input_dir = request.json.get('input_dir')
    output_csv_path = request.json.get('output_csv_path')

    if input_dir is None:
        return jsonify({"status": "error", "message": "Missing required parameter 'input_dir'."}), 400

    stc.extract_audio_info_to_csv(input_dir, output_csv_path)

    return jsonify({"status": "success", "message": f"音频文件成功生成 {input_dir} to {output_csv_path}."})



# 第五步 根据3.数据集生成用户-播放，歌曲-播放数据集
@app.route('/test/play_count')
def playcount():
    return render_template("test/user_play_song_play.html")
@app.route('/generate_playcounts_data', methods=['POST'])
def generate_playcounts_data():
    input_csv_file = request.files.get('play_records_file')
    song_output_csv_file = request.form.get('song_playcounts_file')
    user_output_csv_file = request.form.get('user_playcounts_file')

    # 读取CSV文件
    df = pd.read_csv(io.BytesIO(input_csv_file.read()))

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)

    # 统计歌曲播放次数
    song_output_df = df.groupby('song_id')['play_count'].sum().reset_index()
    sorted_song_df = song_output_df.sort_values(by='play_count', ascending=False)

    # 写入歌曲播放次数到指定CSV文件
    sorted_song_df.to_csv(song_output_csv_file, index=False)

    # 提取用户和播放次数
    df['play_count'] = df['play_count'].astype(int)
    output_df = df.groupby('user_id')['play_count'].sum().reset_index()

    # 将用户按照播放量从高到低排序
    sorted_df = output_df.sort_values(by='play_count', ascending=False)

    # 写入用户播放次数到指定CSV文件
    sorted_df.to_csv(user_output_csv_file, index=False)

    # 返回成功消息
    return jsonify({'message': '生成记录至输出文件目录成功'})
def read_values_from_csv(csv_file_path):
    values_list = []
    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        # 跳过首行（列名）
        next(reader)

        for row in reader:
            values_list.extend(row)

    return values_list
# 第六步 生成完整数据集
@app.route('/test/cpmplete_data')
def complete():
    return render_template("test/complete.html")
@app.route('/generate_complete_dataset', methods=['POST'])
def generate_complete_dataset():
    # 接收请求体中的CSV文件路径
    mp3_csv_path = request.files.get('mp3_csv_path')
    play_records_csv_path = request.files.get('play_records_csv_path')
    output_csv_path = request.form.get('output_csv_path')

    if mp3_csv_path is None or play_records_csv_path is None:
        return jsonify({"status": "error", "message": "Missing required parameter(s): 'mp3_csv_path' and/or 'play_records_csv_path'."}), 400

    try:
        # 加载数据
        mp3_df = pd.read_csv(io.BytesIO(mp3_csv_path.read()))
        play_records_df = pd.read_csv(io.BytesIO(play_records_csv_path.read()))

        # 数据清洗与重命名列
        play_records_df.rename(columns={
            'user_id': 'user_id',
            'song_id': 'song_id',
            'play_count': 'play_count'
        }, inplace=True)

        mp3_df.rename(columns={
            'custom_id': 'song_id',
            'title': 'song',
            'album': 'album'
        }, inplace=True)

        # 合并数据集
        merged_df = pd.merge(mp3_df, play_records_df[['user_id', 'song_id', 'play_count']], on='song_id', how='left')

        # 筛选所需的列
        desired_columns = ['user_id', 'song_id', 'play_count', 'song', 'album', 'artist', 'year']
        merged_df = merged_df[desired_columns]

        # 保存到指定路径
        merged_df.to_csv(output_csv_path, index=False)

        return jsonify({"status": "success", "message": f"数据集成功生成并保存至    {output_csv_path}."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"生成数据时报错: {str(e)}"}), 500

# 进行推荐-协同过滤
@app.route('/test/filter')
def filter():
    return render_template("test/filtering.html")
@app.route('/process_and_recommend', methods=['POST'])
def process_and_recommend():
    # 获取文本参数
    output_file = request.form.get('output_file')  # 对于 'output_file'
    user_id = request.form.get('user_id')

    # 获取文件
    song_playcount_df_file = request.files.get('song_playcount_df')
    user_playcount_df_file = request.files.get('user_playcount_df')
    complete_dataset_file = request.files.get('complete_dataset')
    muisc_df = request.files.get('muisc_df')
    # 检查是否接收到所有必需的文件
    if not all([song_playcount_df_file, user_playcount_df_file, complete_dataset_file]):
        return jsonify(error="Missing one or more required files"), 400
    # 确保文件有效并转换为 pandas DataFrame
    try:
        music_data_df = pd.read_csv(io.BytesIO(muisc_df.read()))
        song_count_df = pd.read_csv(io.BytesIO(song_playcount_df_file.read()))
        play_count_subset = pd.read_csv(io.BytesIO(user_playcount_df_file.read()))
        triplet_dataset_sub_song_merged = pd.read_csv(io.BytesIO(complete_dataset_file.read()))

        song_count_subset = song_count_df.head(n=500)
        user_subset = list(play_count_subset.user_id)
        song_subset = list(song_count_subset.song_id)
        triplet_dataset_sub_song_merged_sub = triplet_dataset_sub_song_merged[
            triplet_dataset_sub_song_merged.song_id.isin(song_subset)]
        train_data, test_data = train_test_split(triplet_dataset_sub_song_merged_sub, test_size=0.30, random_state=0)
        # print("数据集前几条记录:", triplet_dataset_sub_song_merged_sub[:5])

        # 初始化模型
        is_model = coll.item_similarity_recommender_py()
        is_model.create(train_data, 'user_id', 'song_id')

        # 执行推荐
        df_recs = is_model.recommend(user_id)
        df_recs.drop('normalized_score', axis=1, inplace=True)  # 注意添加inplace=True来直接在原DataFrame上操作

        song_ids = df_recs['song_id']
        song_df = music_data_df[['custom_id', 'title', 'artist']]  # 正确提取两列的方式

        # 根据song_ids匹配歌名和歌手
        filtered_songs = song_df[song_df['custom_id'].isin(song_ids)]

        # 更正此处，正确合并'title'和'artist'列到df_recs
        df_recs = df_recs.merge(filtered_songs[['custom_id', 'title', 'artist']],
                                left_on='song_id',
                                right_on='custom_id',
                                how='left')

        # 确保删除重复的'custom_id'列（如果merge操作导致其存在）
        df_recs.drop(columns=['custom_id'], inplace=True, errors='ignore')

        df_recs.rename(columns={
            'user_id': '用户id',
            'song_id': '歌曲编号',
            'title': '歌名',
            'artist': '歌手',
            'score': '推荐得分',
            'rank': '得分排名',
            'scaled_score': '推荐度0-1',
        }, inplace=True)

        # 注意：'title'和'artist'已经在merge时添加，无需在rename中再次提及
        cols = df_recs.columns.tolist()
        last_two_col = cols.pop(-1), cols.pop(-1)
        cols.insert(2, last_two_col[0])
        cols.insert(3, last_two_col[1])
        df_recs = df_recs[cols]
        # 写入Excel，此时df_recs已包含歌名列
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df_recs.to_excel(writer, sheet_name='Sheet1', index=False)
        return jsonify({'message': f'数据excel已经成功保存 {output_file}'})


    except Exception as e:
        return jsonify(error=str(e)), 500  # 返回HTTP状态码500（服务器内部错误）及错误信息


# svd推荐
@app.route('/test/svd')
def svdhtml():
    return render_template("test/svd.html")
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
    triplet_dataset_sub_song_merged = svd.load_data(song_playcount_df_file, user_playcount_df_file, complete_dataset_file)
    recommendations, user_data, small_set = svd.generate_recommendations(user_ids, k, triplet_dataset_sub_song_merged)

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
                    "用户索引": user_idx,
                    "推荐编号": score_idx + 1,  # 推荐编号从1开始计数
                    "歌曲索引": song_index,
                    "歌曲名": song_name,
                    "歌手": artist,
                    "推荐度(0-1)": round(normalized_score, 4)  # 保留四位小数
                })

        # 将列表转换为DataFrame
        df_recommendations = pd.DataFrame(recommendation_list)
        df_recommendations.to_excel(output_file, index=False)

    return jsonify({'message': f'数据成功写入excel {output_file}'})


if __name__ == '__main__':
    app.run()
    print(current_app.template_folder)