import os
import pandas as pd
from collections import defaultdict

def create_small_dataset(input_file, output_file, sampling_rate=0.05):
    """读取TSV文件，只保留前N%的claim_id对应的所有数据"""
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取TSV数据
    df = pd.read_csv(input_file, sep='\t')
    print(f"原始数据: {input_file}, 行数: {len(df)}")
    
    # 获取所有唯一的claim_id
    unique_claims = df['claim_id'].unique()
    
    # 计算要保留的claim_id数量
    keep_count = max(1, int(len(unique_claims) * sampling_rate))
    
    # 选择前N%的claim_id
    selected_claims = unique_claims[:keep_count]
    
    # 只保留这些claim_id的数据
    result_df = df[df['claim_id'].isin(selected_claims)]
    
    print(f"保留的claim_id数量: {len(selected_claims)}/{len(unique_claims)}")
    print(f"抽样后数据行数: {len(result_df)}/{len(df)}")
    
    # 保存结果
    result_df.to_csv(output_file, sep='\t', index=False)
    print(f"数据已保存到: {output_file}")

def process_dataset(base_path):
    """处理指定目录下的所有数据集文件"""
    # 创建输出目录
    output_base = os.path.join(os.path.dirname(base_path), "split_data")
    os.makedirs(output_base, exist_ok=True)
    
    # 处理所有fold目录
    for fold_dir in os.listdir(base_path):
        fold_path = os.path.join(base_path, fold_dir)
        if os.path.isdir(fold_path):
            output_fold_path = os.path.join(output_base, fold_dir)
            os.makedirs(output_fold_path, exist_ok=True)
            
            # 处理训练和测试文件
            for file_name in os.listdir(fold_path):
                if file_name.startswith(('train_', 'test_')) and file_name.endswith('.tsv'):
                    input_file = os.path.join(fold_path, file_name)
                    output_file = os.path.join(output_fold_path, file_name)
                    create_small_dataset(input_file, output_file)

if __name__ == "__main__":
    # 数据集基本路径
    base_path = "formatted_data/declare/PolitiFact/mapped_data"
    process_dataset(base_path)
    
    # 单独处理dev.tsv
    create_small_dataset("formatted_data/declare/PolitiFact/mapped_data/dev.tsv",
                        "formatted_data/declare/PolitiFact/split_data/dev.tsv",
                        sampling_rate=0.05)
    
    print("所有数据处理完成。")