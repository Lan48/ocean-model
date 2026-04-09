import os
import shutil

def copy_files_by_date_range(source_dir, target_dir, start_date, end_date):
    """
    根据日期范围复制文件：从源目录复制文件名符合"YYYY_MM.npy"格式且日期在范围内的文件到目标目录。
    
    参数:
        source_dir (str): 源目录路径
        target_dir (str): 目标目录路径
        start_date (str): 起始日期（格式为"YYYY-MM"，例如"2023-01"）
        end_date (str): 结束日期（格式同起始日期）
    """
    # 检查目标目录是否存在，若不存在则创建[6,7](@ref)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")
    
    # 遍历源目录及其所有子目录[1,5](@ref)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名是否为.npy
            if file.endswith('.npy'):
                # 解析文件名：提取年份和月份
                base_name = os.path.splitext(file)[0]  # 去掉扩展名，得到"YYYY_MM"
                parts = base_name.split('_')  # 按"_"分割为[年份, 月份]
                
                # 验证文件名格式是否正确（两部分且均为数字）
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    year = parts[0]
                    month = parts[1]
                    file_date = f"{year}-{month}"  # 组合成"YYYY-MM"格式，便于比较
                    
                    # 检查文件日期是否在指定范围内[2](@ref)
                    if start_date <= file_date <= end_date:
                        # 构建完整路径
                        source_path = os.path.join(root, file)
                        target_path = os.path.join(target_dir, file)
                        
                        try:
                            # 复制文件（保留元数据，如修改时间）[6,7](@ref)
                            shutil.copy2(source_path, target_path)
                            print(f"已复制: {file} 从 {source_path} 到 {target_path}")
                        except Exception as e:
                            print(f"复制失败 {file}: {e}")
                else:
                    print(f"文件名格式错误，跳过: {file}")
    print("文件复制操作完成。")

# 使用示例
if __name__ == "__main__":
    source = input("请输入源目录路径: ")
    target = input("请输入目标目录路径: ")
    start = input("请输入起始日期（格式: YYYY-MM）: ")
    end = input("请输入结束日期（格式: YYYY-MM）: ")
    copy_files_by_date_range(source, target, start, end)