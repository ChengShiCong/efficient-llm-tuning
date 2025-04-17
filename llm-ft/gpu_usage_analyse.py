import csv

file_path = "/root/autodl-tmp/ft_adapter/gpu_usage/deepseek-distill-1.5B-20250412_2255/gpu_usage.csv"

used_memory_list = []
total_memory_list = []
utilization_percent_list = []

try:
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并跳过 header

        for row in reader:
            try:
                used_memory_list.append(float(row[2]))
                total_memory_list.append(float(row[3]))
                utilization_percent_list.append(int(row[4]))
            except (ValueError, IndexError) as e:
                print(f"解析行时出错: {row} - {e}")

except FileNotFoundError:
    print(f"错误：文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时发生错误: {e}")

# 计算平均值
avg_used_memory = sum(used_memory_list) / len(used_memory_list) if used_memory_list else 0
avg_total_memory = sum(total_memory_list) / len(total_memory_list) if total_memory_list else 0
avg_utilization_percent = sum(utilization_percent_list) / len(utilization_percent_list) if utilization_percent_list else 0

# 打印结果
print(f"平均使用的显存 (used_memory_MB): {avg_used_memory:.2f}")
print(f"平均总显存 (total_memory_MB): {avg_total_memory:.2f}")
print(f"平均 GPU 利用率 (utilization_percent): {avg_utilization_percent:.2f}%")