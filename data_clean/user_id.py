#当自定义id为空时，进行批量补偿
import os
from mutagen.id3 import ID3
def update_txxx_description(directory, target_desc):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                filepath = os.path.join(root, file)

                # 打开MP3文件的ID3标签
                audio = ID3(filepath)

                # 遍历所有TXXX帧，寻找那些文本信息不是空的且描述为空的帧
                frames_to_update = [frame for frame in audio.getall('TXXX')
                                    if frame.desc is None or frame.desc == ''
                                    and frame.text[0] != '']

                # 更新这些帧的描述
                for frame in frames_to_update:
                    frame.desc = target_desc

                # 保存更改
                audio.save(v2_version=3)

# 使用示例
#target_directory = 'E:/music/华语歌曲/'
#target_description = 'Custom ID'