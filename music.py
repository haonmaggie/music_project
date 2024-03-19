import os
import glob
import pandas as pd
from mutagen.id3 import ID3

def get_mp3_metadata(file_path):
    audio = ID3(file_path)
    metadata = {
        'title': audio.get('TIT2', [''])[0].text[0] if 'TIT2' in audio else '',
        'artist': audio.get('TPE1', [''])[0].text[0] if 'TPE1' in audio else '',
        # 其他元数据...
        'file_path': file_path,
    }
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

# 将数据集转换为Pandas DataFrame（如果需要进一步分析或处理）
df = pd.DataFrame(music_dataset)

# 若有年份信息，按年份分类等后续操作...