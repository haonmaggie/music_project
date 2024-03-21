import os
from mutagen.id3 import ID3


def get_text_from_id3_tag(audio, tag):
    tag_info = audio.get(tag)

    if tag_info and isinstance(tag_info, list) and len(tag_info) > 0 and hasattr(tag_info[0], 'text') and tag_info[
        0].text:
        return tag_info[0].text[0]
    else:
        return ''
def get_mp3_metadata(file_path):
    audio = ID3(file_path)

    title = get_text_from_id3_tag(audio, 'TIT2')
    artist = get_text_from_id3_tag(audio, 'TPE1')
    album = get_text_from_id3_tag(audio, 'TALB')
    track_number = get_text_from_id3_tag(audio, 'TRCK')
    year = get_text_from_id3_tag(audio, 'TDRC')
    composer = get_text_from_id3_tag(audio,'TCOM')
    album_artist = get_text_from_id3_tag(audio,'TPE2')
    duration = audio.get('TLEN', [0])[0]  # 如果存在，会返回一个整数，表示毫秒数

    # 将时长转换为分钟和秒（假设已存在）
    if duration:
        minutes, seconds = divmod(duration // 1000, 60)
        duration_text = f"{minutes:02d}:{seconds:02d}"
    else:
        duration_text = ''
    metadata = {
        'title': title,
        'artist': artist,
        'album': album,
        'track_number': track_number,
        'year': year,
        'composer': composer,
        'album_artist': album_artist,
        'duration': duration_text,
        'file_path': file_path,
    }
    return metadata

def build_music_dataset(root_folder, dataset=None):
    if dataset is None:
        dataset = []

    for artist_folder_name in os.listdir(root_folder):
        artist_folder = os.path.join(root_folder, artist_folder_name)
        if os.path.isdir(artist_folder):
            # 省略了对二级目录的遍历，直接在一级目录下查找MP3文件
            for mp3_file in os.listdir(artist_folder):
                if mp3_file.endswith('.mp3'):
                    mp3_path = os.path.join(artist_folder, mp3_file)
                    metadata = get_mp3_metadata(mp3_path)
                    dataset.append(metadata)

    return dataset

# 使用方法：
root_folder = r'G:\1200首华语歌坛精选（131位歌手）- 推荐'
music_dataset = build_music_dataset(root_folder)

# 将数据集转换为Pandas DataFrame以便后续处理或分析
import pandas as pd
df = pd.DataFrame(music_dataset)
output_directory = 'E:/music/'
filename = '音乐数据集.csv'
# 创建完整输出路径
output_path = os.path.join(output_directory, filename)
# 确保目标目录存在，如果不存在则创建它
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 将DataFrame保存为HDF5文件
df.to_csv('音乐数据集.csv', index=False)