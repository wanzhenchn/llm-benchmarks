import json
import csv
import os
import argparse
from pathlib import Path

def load_json_data(file_path):
    """加载JSON文件数据（处理每行一个JSON对象的情况）"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data_list.append(json.loads(line))
    return data_list

def extract_config_label(file_path):
    """从文件名提取配置标签"""
    filename = os.path.basename(file_path)

    # 使用 split("-")[-1].split(".json")[0] 提取标签
    try:
        label = filename.split("-")[-1].split(".json")[0]
        return label
    except:
        # 如果提取失败，返回文件名（不含扩展名）
        return os.path.splitext(filename)[0]

def calculate_service_throughput(duration, completed):
    """计算服务吞吐量 (req/s)"""
    if duration > 0 and completed > 0:
        return completed / duration
    return 0

def convert_json_to_csv(json_data, output_file):
    """将JSON数据转换为CSV格式"""

    # CSV文件头
    headers = [
        'Successful Request',
        'Request_Gen_Token_Len',
        'Batch Size',
        'Avg_Input_Token_Len',
        'Avg_Gen_Token_Len',
        'Elapse_Time (s)',
        'Time_to_First_Token_AVG (ms)',
        'Time_to_First_Token_P99 (ms)',
        'Time_per_Output_Token_AVG (ms)',
        'Time_per_Output_Token_P99 (ms)',
        'Token Throughput (token/s)',
        'Service Throughput (req/s)'
    ]

    # 准备数据行
    rows = []

    for data in json_data:
        # 计算需要的字段
        successful_request = data.get('completed', 0)
        request_gen_token_len = 100  # 固定值
        batch_size = data.get('max_concurrency', 0)

        # 计算平均输入token长度
        total_input_tokens = data.get('total_input_tokens', 0)
        avg_input_token_len = total_input_tokens / successful_request if successful_request > 0 else 0

        # 计算平均生成token长度
        total_output_tokens = data.get('total_output_tokens', 0)
        avg_gen_token_len = total_output_tokens / successful_request if successful_request > 0 else 0

        elapse_time = data.get('duration', 0)
        time_to_first_token_avg = data.get('mean_ttft_ms', 0)
        time_to_first_token_p99 = data.get('p99_ttft_ms', 0)
        time_per_output_token_avg = data.get('mean_tpot_ms', 0)
        time_per_output_token_p99 = data.get('p99_tpot_ms', 0)
        token_throughput = data.get('output_throughput', 0)
        service_throughput = calculate_service_throughput(elapse_time, successful_request)

        # 创建数据行
        row = [
            successful_request,
            request_gen_token_len,
            batch_size,
            round(avg_input_token_len, 2),
            round(avg_gen_token_len, 2),
            round(elapse_time, 3),
            round(time_to_first_token_avg, 2),
            round(time_to_first_token_p99, 2),
            round(time_per_output_token_avg, 3),
            round(time_per_output_token_p99, 3),
            round(token_throughput, 2),
            round(service_throughput, 4)
        ]

        rows.append(row)

    # 写入CSV文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(headers)

        # 写入数据行
        writer.writerows(rows)

    print(f"CSV文件已保存为: {output_file}")
    print(f"共转换 {len(rows)} 行数据")

    return rows

def display_csv_preview(rows, num_rows=5):
    """显示CSV数据预览"""
    if not rows:
        print("没有数据可显示")
        return

    print("\nCSV数据预览 (前{}行):".format(min(num_rows, len(rows))))
    print("-" * 120)

    # 显示表头
    headers = [
        'Successful Request',
        'Request_Gen_Token_Len',
        'Batch Size',
        'Avg_Input_Token_Len',
        'Avg_Gen_Token_Len',
        'Elapse_Time (s)',
        'Time_to_First_Token_AVG (ms)',
        'Time_to_First_Token_P99 (ms)',
        'Time_per_Output_Token_AVG (ms)',
        'Time_per_Output_Token_P99 (ms)',
        'Token Throughput (token/s)',
        'Service Throughput (req/s)'
    ]

    # 格式化显示表头
    header_format = "{:>18} {:>22} {:>10} {:>18} {:>17} {:>15} {:>28} {:>28} {:>30} {:>30} {:>25} {:>22}"
    print(header_format.format(*headers[:12]))
    print("-" * 120)

    # 显示数据行
    for i, row in enumerate(rows[:num_rows]):
        formatted_row = [
            str(row[0]),
            str(row[1]),
            str(row[2]),
            f"{row[3]:.2f}",
            f"{row[4]:.2f}",
            f"{row[5]:.3f}",
            f"{row[6]:.2f}",
            f"{row[7]:.2f}",
            f"{row[8]:.3f}",
            f"{row[9]:.3f}",
            f"{row[10]:.2f}",
            f"{row[11]:.4f}"
        ]
        print(header_format.format(*formatted_row))

def main():
    parser = argparse.ArgumentParser(description='将JSON性能数据转换为CSV格式')
    parser.add_argument('json_file', help='JSON文件路径')
    parser.add_argument('-o', '--output', help='输出CSV文件名（可选，默认为JSON文件名+.csv）')
    parser.add_argument('-p', '--preview', type=int, default=5,
                       help='预览行数（默认5行，0表示不预览）')

    args = parser.parse_args()

    # 加载JSON数据
    json_data = load_json_data(args.json_file)

    # 设置输出文件名
    if args.output:
        output_file = args.output
    else:
        # 使用JSON文件名生成CSV文件名
        base_name = os.path.splitext(args.json_file)[0]
        output_file = f"{base_name}.csv"

    # 从文件名提取配置标签
    config_label = extract_config_label(args.json_file)

    # 转换数据
    rows = convert_json_to_csv(json_data, output_file)

    # 显示预览
    if args.preview > 0 and rows:
        display_csv_preview(rows, args.preview)

    # 显示统计信息
    if rows:
        print("\n数据统计:")
        print("-" * 50)

        # 计算关键指标的统计值
        batch_sizes = [row[2] for row in rows]
        token_throughputs = [row[10] for row in rows]
        service_throughputs = [row[11] for row in rows]
        ttft_avgs = [row[6] for row in rows]

        print(f"并发度范围: {min(batch_sizes)} - {max(batch_sizes)}")
        print(f"最大Token吞吐量: {max(token_throughputs):.2f} tokens/s (Batch Size: {batch_sizes[token_throughputs.index(max(token_throughputs))]})")
        print(f"最大服务吞吐量: {max(service_throughputs):.4f} req/s (Batch Size: {batch_sizes[service_throughputs.index(max(service_throughputs))]})")
        print(f"最小TTFT: {min(ttft_avgs):.2f} ms (Batch Size: {batch_sizes[ttft_avgs.index(min(ttft_avgs))]})")

def batch_convert_multiple_files():
    """批量转换多个JSON文件"""
    parser = argparse.ArgumentParser(description='批量将多个JSON性能数据转换为CSV格式')
    parser.add_argument('json_files', nargs='+', help='JSON文件路径（支持通配符）')
    parser.add_argument('-p', '--preview', type=int, default=3,
                       help='每个文件的预览行数（默认3行，0表示不预览）')

    args = parser.parse_args()

    for json_file in args.json_files:
        print("\n" + "="*60)
        print(f"处理文件: {json_file}")
        print("="*60)

        try:
            # 加载JSON数据
            json_data = load_json_data(json_file)

            if not json_data:
                print(f"警告: 文件 {json_file} 没有数据或格式不正确")
                continue

            # 设置输出文件名
            base_name = os.path.splitext(json_file)[0]
            output_file = f"{base_name}.csv"

            # 从文件名提取配置标签
            config_label = extract_config_label(json_file)

            # 转换数据
            rows = convert_json_to_csv(json_data, output_file)

            # 显示预览
            if args.preview > 0 and rows:
                display_csv_preview(rows, min(args.preview, len(rows)))

        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")

if __name__ == "__main__":
    # 使用示例:
    # python json_to_csv.py perf-Qwen2.5-14B-Instruct-NUM1000-ISL7500-OSL200-TP4.json
    # python json_to_csv.py perf-Qwen2.5-14B-Instruct-NUM1000-ISL7500-OSL200-TP4.json -o tp4_performance.csv
    # python json_to_csv.py perf-Qwen2.5-14B-Instruct-NUM1000-ISL7500-OSL200-TP4.json -p 10

    main()

    # 如果要批量转换多个文件，可以取消下面的注释并注释掉上面的 main() 调用
    # batch_convert_multiple_files()
