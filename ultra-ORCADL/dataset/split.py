import os
from collections import defaultdict
import shutil

def keep_files_in_all_subdirs(directory):
    """
    保留仅在所有子目录中都存在同名文件的文件，删除其他所有文件。
    
    参数:
        directory (str): 要处理的目录路径
    """
    # 获取所有一级子目录
    subdirs = [d for d in os.listdir(directory) 
               if os.path.isdir(os.path.join(directory, d))]
    
    if not subdirs:
        print("该目录下没有子目录。")
        return
    
    print(f"找到 {len(subdirs)} 个子目录: {subdirs}")
    
    # 使用字典存储文件名和它出现在哪些子目录中
    file_presence_dict = defaultdict(set)
    
    # 遍历每个子目录，记录文件出现情况
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        for root, dirs, files in os.walk(subdir_path):
            for file in files:
                file_presence_dict[file].add(subdir)
    
    # 找出在所有子目录中都存在的文件
    files_in_all_subdirs = []
    total_subdirs = len(subdirs)
    
    for filename, present_in_subdirs in file_presence_dict.items():
        if len(present_in_subdirs) == total_subdirs:
            files_in_all_subdirs.append(filename)
    
    if not files_in_all_subdirs:
        print("没有找到在所有子目录中都存在的同名文件。")
        return
    
    print(f"\n找到 {len(files_in_all_subdirs)} 个在所有子目录中都存在的文件:")
    for file in files_in_all_subdirs:
        print(f"  {file}")
    
    # 收集所有文件路径，分为保留和删除两类
    files_to_keep = []
    files_to_delete = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file in files_in_all_subdirs:
                files_to_keep.append(file_path)
            else:
                files_to_delete.append(file_path)
    
    # 输出结果
    print("\n=== 保留的文件（在所有子目录中都存在）===")
    for path in files_to_keep:
        print(f"保留: {path}")
    
    print("\n=== 待删除的文件 ===")
    for path in files_to_delete:
        print(f"删除: {path}")
    
    # 确认后执行删除操作
    if files_to_delete:
        confirm = input(f"\n是否确认删除 {len(files_to_delete)} 个文件？(y/n): ")
        if confirm.lower() == 'y':
            for path in files_to_delete:
                try:
                    os.remove(path)
                    print(f"已删除: {path}")
                except Exception as e:
                    print(f"删除失败 {path}: {e}")
        else:
            print("操作已取消。")
    else:
        print("没有需要删除的文件。")

# 使用示例
if __name__ == "__main__":
    target_directory = input("请输入目录路径: ")
    keep_files_in_all_subdirs(target_directory)