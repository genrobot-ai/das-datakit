import os
import json

def load_txt(txt_file_path):
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            filenames = [line.strip() for line in file.readlines()]
            filenames = [name for name in filenames if name]
        return filenames
    except FileNotFoundError:
        print(f"错误：找不到文件 {txt_file_path}")
        return []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return []
    
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data