

import os
import shutil
import zipfile
import chardet


def get_encoding(path):
    # 读取文件的前几行，检测编码
    with open(path, 'rb') as f:
        result = chardet.detect(f.read(1000))
        encoding = result['encoding']
        return encoding

def read_all(path: str) -> str:
    root_dir = get_root_dir()
    path = os.path.join(root_dir, path)
    # Open the file in read mode
    file = open(path, 'r', encoding='utf-8')
    # Read the content of the file as a string
    content = file.read()
    # Close the file
    file.close()
    return content

def get_root_dir():
    # 方法一 从其他目录调用该方法不一定返回正确的目录

    # current_dir = os.getcwd() # Get the current directory
    # root_dir = os.path.dirname(current_dir)
    # 方法二
    root_dir = os.path.dirname(os.path.abspath(__file__))
    #drop folder name 'utils/file'
    root_dir = root_dir[:-11]
    return root_dir

def write_to_file(path,content):
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Open file in write mode; if file doesn't exist, it will be created
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        print(f"Successfully written to '{file}'.")

    except Exception as e:
        print(f"Error occurred: {e}")


def compress_folder(path1):
    # 检查 path1 文件夹是否存在
    if not os.path.exists(path1):
        print(f"文件夹 {path1} 不存在")
        return

    # 创建压缩文件
    zip_file_name = "compress.zip"
    zip_path = os.path.join(path1, zip_file_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # 遍历 path1 文件夹下的所有文件和文件夹，逐个添加到压缩文件中
        for root, dirs, files in os.walk(path1):
            for file in files:
                file_path = os.path.join(root, file)
                # 在压缩文件中创建相同的目录结构
                zip_file.write(file_path, os.path.relpath(file_path, path1))

    return zip_path, zip_file_name


def copy(path1,path2,file_name):
    # 检查 path2 文件夹是否存在，不存在则创建
    if not os.path.exists(path2):
        os.makedirs(path2)

    # 复制压缩文件到 path2
    shutil.copy(path1, os.path.join(path2, file_name))
    print(f"成功将文件夹 {path1} 下的文件打包为 {file_name} 并复制到 {path2}")




def find_files_with_suffix(directory, suffix=None):

    # 存储符合条件的文件名
    matching_files = []

    # 使用 os.walk 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否以指定后缀结尾
            if suffix is None or file.endswith(suffix):
                # 将符合条件的文件路径加入列表
                matching_files.append(os.path.join(root, file))

    return matching_files

def convert(data):
    import json
    data = json.loads(data)
    content = ''
    for i, code in enumerate(data['exp_codes']):
        print(f"Processed exp_code {i + 1}:")
        for line in code.split('\n'):
            print(line)
            content += line
            content += '\n'
    return content

import re


def replace_multiple_words(text, replacements):
    # 构建正则表达式模式，匹配所有需要替换的独立单词
    pattern = r'\b(' + '|'.join(re.escape(word) for word in replacements.keys()) + r')\b'

    # 定义替换函数
    def replacer(match):
        # 返回匹配的单词在字典中的替换值
        return replacements[match.group(0)]

    # 使用 re.sub 进行替换
    replaced_text = re.sub(pattern, replacer, text)
    return replaced_text




if __name__ == '__main__':
    # 调用示例
    # path1 = "d:/test"
    # path2 = "e:/"
    # zip_path,zip_file_name = compress_folder(path1)
    # copy(zip_path,path2,zip_file_name)

    # content = read_all('data/circuits/xeb/xeb3/XEB_3_qubits_8_cycles_circuit.txt')
    # print(content)

    #
    # dir_path = r'D:\workspace\电信大赛\3_66_ghz_circuit\3_66_ghz_circuit\3_66_ghz_circuit'
    # files = find_files_with_suffix(dir_path)
    #
    # for file in files:
    #     content = read_all(file)
    #     new_content = convert(content)
    #     write(file.replace('.json','.txt'),new_content)

    # 示例用法
    # text = read_all(r'D:\workspace\电信大赛\3_66_ghz_circuit\3_66_ghz_circuit\3_66_ghz_circuit\3qubit_ghz.txt')
    #
    # replacements = {
    #     "Q0": "Q21",
    #     "Q1": "Q26",
    #     "Q2": "Q33"
    # }
    # result = replace_multiple_words(text,replacements)
    # print(result)
    print(get_root_dir())





