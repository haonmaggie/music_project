import os
from mutagen.id3 import ID3


def get_mp3_metadata(file_path):
    audio = ID3(file_path)
    metadata = {
        'title': audio.get('TIT2', [''])[0].text[0] if 'TIT2' in audio else '',
        'artist': audio.get('TPE1', [''])[0].text[0] if 'TPE1' in audio else '',
        # 如果需要年份等其他信息，可以从ID3标签或其他来源获取
        'file_name_artist_song': os.path.splitext(os.path.basename(file_path))[0],
    }

    # 提取文件名中的歌手名和歌曲名（假设它们由空格分隔）
    artist_song = metadata['file_name_artist_song'].split(' ')
    metadata['artist_from_filename'] = artist_song[0]
    metadata['song_from_filename'] = ' '.join(artist_song[1:])

    return metadata


def build_music_dataset(root_folder, dataset=None):
    if dataset is None:
        dataset = []

    for entry in os.scandir(root_folder):
        if entry.is_dir():
            build_music_dataset(entry.path, dataset)
        elif entry.name.endswith('.mp3'):
            metadata = get_mp3_metadata(entry.path)
            dataset.append(metadata)

    return dataset


# 使用方法：
root_folder = '华语歌曲'  # 或任何其他可能的根目录
music_dataset = build_music_dataset(root_folder)

# 将数据集转换为Pandas DataFrame以便后续处理或分析
import pandas as pd

df = pd.DataFrame(music_dataset)

# 若需按年份分类，首先确保你已从元数据或其他途径获得年份信息
# df['year'] = ...（填充年份信息）
# 然后可以按年份排序或分组