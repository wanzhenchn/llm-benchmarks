import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse

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

    try:
        label = filename.split("-")[-1].split(".json")[0]
        return label
    except:
        # 如果提取失败，返回文件名（不含扩展名）
        return os.path.splitext(filename)[0]

def extract_metrics(data_list):
    """从数据列表中提取指定指标"""
    # 按max_concurrency排序
    data_list.sort(key=lambda x: x['max_concurrency'])

    metrics = {
        'max_concurrency': [],
        'mean_ttft_ms': [],
        'p99_ttft_ms': [],
        'mean_tpot_ms': [],
        'p99_tpot_ms': [],
        'output_throughput': [],
    }

    for data in data_list:
        metrics['max_concurrency'].append(data['max_concurrency'])
        metrics['mean_ttft_ms'].append(data['mean_ttft_ms'])
        metrics['p99_ttft_ms'].append(data['p99_ttft_ms'])
        metrics['mean_tpot_ms'].append(data['mean_tpot_ms'])
        metrics['p99_tpot_ms'].append(data['p99_tpot_ms'])
        metrics['output_throughput'].append(data['output_throughput'])

    return metrics

def calculate_ratio_arrays(baseline_values, disagg_values, baseline_x, disagg_x):
    """计算两个数组的比值，对齐相同并发度的数据"""
    ratio_dict = {}

    for i, x in enumerate(baseline_x):
        if x in disagg_x:
            disagg_idx = disagg_x.index(x)
            if baseline_values[i] != 0:  # 避免除以0
                ratio = disagg_values[disagg_idx] / baseline_values[i]
                ratio_dict[x] = ratio

    # 转换为对齐的数组
    common_x = sorted(ratio_dict.keys())
    ratios = [ratio_dict[x] for x in common_x]

    return common_x, ratios

def calculate_ratio(value1, value2):
    """计算单个值的比值"""
    if value1 != 0:
        return value2 / value1
    return 0

def create_plot(baseline_data,
                disagg_data,
                baseline_label,
                disagg_label,
                title: str = "Perf of disaggregated PD",
                subtitle: str = "ISL/OSL",
                output_file='perf_disagg.png'):
    """创建双Y轴对比图"""

    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # 添加主标题和副标题
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=14, color='#555555')

    # 提取数据
    baseline_metrics = extract_metrics(baseline_data)
    disagg_metrics = extract_metrics(disagg_data)

    # 获取并发度列表
    baseline_x = baseline_metrics['max_concurrency']
    disagg_x = disagg_metrics['max_concurrency']

    # 设置统一的X轴刻度
    all_concurrencies = sorted(set(baseline_x + disagg_x))

    # 定义颜色
    baseline_color = '#1f77b4'  # 蓝色
    disagg_color = '#ff7f0e'    # 橙色
    ratio_color = '#d62728'     # 红色

    # 1. Mean TTFT对比图（双Y轴）
    ax1 = axes[0, 0]
    ax1_ratio = ax1.twinx()

    # 绘制原始数据
    ax1.plot(baseline_x, baseline_metrics['mean_ttft_ms'], 'o-', linewidth=2, markersize=8,
             label=baseline_label, color=baseline_color)
    ax1.plot(disagg_x, disagg_metrics['mean_ttft_ms'], 's-', linewidth=2, markersize=8,
             label=disagg_label, color=disagg_color)

    # 计算并绘制比值
    ratio_x, ratio_values = calculate_ratio_arrays(
        baseline_metrics['mean_ttft_ms'], disagg_metrics['mean_ttft_ms'],
        baseline_x, disagg_x
    )
    ax1_ratio.plot(ratio_x, ratio_values, '^--', linewidth=1.5, markersize=6,
                   label=f'{disagg_label}/{baseline_label}', color=ratio_color, alpha=0.8)

    # 添加ratio数据标注
    for x, y in zip(ratio_x, ratio_values):
        ax1_ratio.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=8, color=ratio_color, fontweight='bold')

    ax1.set_xlabel('Max Concurrency')
    ax1.set_ylabel('Mean TTFT (ms)', color=baseline_color)
    ax1_ratio.set_ylabel('Ratio', color=ratio_color)
    ax1.set_title('Mean TTFT')  # 简化标题
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_xticks(all_concurrencies)
    ax1.set_xticklabels([str(x) for x in all_concurrencies])

    # 设置颜色
    ax1.tick_params(axis='y', labelcolor=baseline_color)
    ax1_ratio.tick_params(axis='y', labelcolor=ratio_color)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_ratio.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 2. P99 TTFT对比图（双Y轴）
    ax2 = axes[0, 1]
    ax2_ratio = ax2.twinx()

    ax2.plot(baseline_x, baseline_metrics['p99_ttft_ms'], 'o-', linewidth=2, markersize=8,
             label=baseline_label, color=baseline_color)
    ax2.plot(disagg_x, disagg_metrics['p99_ttft_ms'], 's-', linewidth=2, markersize=8,
             label=disagg_label, color=disagg_color)

    ratio_x, ratio_values = calculate_ratio_arrays(
        baseline_metrics['p99_ttft_ms'], disagg_metrics['p99_ttft_ms'],
        baseline_x, disagg_x
    )
    ax2_ratio.plot(ratio_x, ratio_values, '^--', linewidth=1.5, markersize=6,
                   label=f'{disagg_label}/{baseline_label}', color=ratio_color, alpha=0.8)
    # 添加ratio数据标注
    for x, y in zip(ratio_x, ratio_values):
        ax2_ratio.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=8, color=ratio_color, fontweight='bold')

    ax2.set_xlabel('Max Concurrency')
    ax2.set_ylabel('P99 TTFT (ms)', color=baseline_color)
    ax2_ratio.set_ylabel('Ratio', color=ratio_color)
    ax2.set_title('P99 TTFT')  # 简化标题
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_xticks(all_concurrencies)
    ax2.set_xticklabels([str(x) for x in all_concurrencies])

    ax2.tick_params(axis='y', labelcolor=baseline_color)
    ax2_ratio.tick_params(axis='y', labelcolor=ratio_color)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_ratio.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 3. Mean TPOT对比图（双Y轴）- 调整到第2行
    ax3 = axes[1, 0]
    ax3_ratio = ax3.twinx()

    ax3.plot(baseline_x, baseline_metrics['mean_tpot_ms'], 'o-', linewidth=2, markersize=8,
             label=baseline_label, color=baseline_color)
    ax3.plot(disagg_x, disagg_metrics['mean_tpot_ms'], 's-', linewidth=2, markersize=8,
             label=disagg_label, color=disagg_color)

    ratio_x, ratio_values = calculate_ratio_arrays(
        baseline_metrics['mean_tpot_ms'], disagg_metrics['mean_tpot_ms'],
        baseline_x, disagg_x
    )
    ax3_ratio.plot(ratio_x, ratio_values, '^--', linewidth=1.5, markersize=6,
                   label=f'{disagg_label}/{baseline_label}', color=ratio_color, alpha=0.8)
    # 添加ratio数据标注
    for x, y in zip(ratio_x, ratio_values):
        ax3_ratio.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=8, color=ratio_color, fontweight='bold')

    ax3.set_xlabel('Max Concurrency')
    ax3.set_ylabel('Mean TPOT (ms)', color=baseline_color)
    ax3_ratio.set_ylabel('Ratio', color=ratio_color)
    ax3.set_title('Mean TPOT')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_xticks(all_concurrencies)
    ax3.set_xticklabels([str(x) for x in all_concurrencies])

    ax3.tick_params(axis='y', labelcolor=baseline_color)
    ax3_ratio.tick_params(axis='y', labelcolor=ratio_color)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_ratio.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 4. P99 TPOT对比图（双Y轴)
    ax4 = axes[1, 1]
    ax4_ratio = ax4.twinx()

    ax4.plot(baseline_x, baseline_metrics['p99_tpot_ms'], 'o-', linewidth=2, markersize=8,
             label=baseline_label, color=baseline_color)
    ax4.plot(disagg_x, disagg_metrics['p99_tpot_ms'], 's-', linewidth=2, markersize=8,
             label=disagg_label, color=disagg_color)

    ratio_x, ratio_values = calculate_ratio_arrays(
        baseline_metrics['p99_tpot_ms'], disagg_metrics['p99_tpot_ms'],
        baseline_x, disagg_x
    )
    ax4_ratio.plot(ratio_x, ratio_values, '^--', linewidth=1.5, markersize=6,
                   label=f'{disagg_label}/{baseline_label}', color=ratio_color, alpha=0.8)
    # 添加ratio数据标注
    for x, y in zip(ratio_x, ratio_values):
        ax4_ratio.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=8, color=ratio_color, fontweight='bold')

    ax4.set_xlabel('Max Concurrency')
    ax4.set_ylabel('P99 TPOT (ms)', color=baseline_color)
    ax4_ratio.set_ylabel('Ratio', color=ratio_color)
    ax4.set_title('P99 TPOT')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_xticks(all_concurrencies)
    ax4.set_xticklabels([str(x) for x in all_concurrencies])

    ax4.tick_params(axis='y', labelcolor=baseline_color)
    ax4_ratio.tick_params(axis='y', labelcolor=ratio_color)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_ratio.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 5. Output Throughput对比图（双Y轴)
    ax5 = axes[2, 0]
    ax5_ratio = ax5.twinx()

    ax5.plot(baseline_x, baseline_metrics['output_throughput'], 'o-', linewidth=2, markersize=8,
             label=baseline_label, color=baseline_color)
    ax5.plot(disagg_x, disagg_metrics['output_throughput'], 's-', linewidth=2, markersize=8,
             label=disagg_label, color=disagg_color)

    ratio_x, ratio_values = calculate_ratio_arrays(
        baseline_metrics['output_throughput'], disagg_metrics['output_throughput'],
        baseline_x, disagg_x
    )
    ax5_ratio.plot(ratio_x, ratio_values, '^--', linewidth=1.5, markersize=6,
                   label=f'{disagg_label}/{baseline_label}', color=ratio_color, alpha=0.8)
    # 添加ratio数据标注
    for x, y in zip(ratio_x, ratio_values):
        ax5_ratio.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10),
                          textcoords='offset points', ha='center', va='bottom',
                          fontsize=8, color=ratio_color, fontweight='bold')

    ax5.set_xlabel('Max Concurrency')
    ax5.set_ylabel('TPS (tokens/s)', color=baseline_color)
    ax5_ratio.set_ylabel('Ratio', color=ratio_color)
    ax5.set_title('Output Token Throughput')
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_xticks(all_concurrencies)
    ax5.set_xticklabels([str(x) for x in all_concurrencies])

    ax5.tick_params(axis='y', labelcolor=baseline_color)
    ax5_ratio.tick_params(axis='y', labelcolor=ratio_color)

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_ratio.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax6 = axes[2, 1]
    ax6.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"图表已保存为: {output_file}")

    # 输出详细的原始数据表格（按照之前的版本，但在disagg数据表中右侧新增对应metrics比值）
    print("\n" + "="*90)
    print(f"{baseline_label} 数据:")
    print("="*90)
    print(f"{'Concurrency':>12} {'Mean TTFT':>12} {'P99 TTFT':>12} {'TPS':>11} {'Mean TPOT':>12} {'P99 TPOT':>12}")
    print("-"*90)
    for i, x in enumerate(baseline_x):
        print(f"{x:>12} {baseline_metrics['mean_ttft_ms'][i]:>11.1f} {baseline_metrics['p99_ttft_ms'][i]:>11.1f} {baseline_metrics['output_throughput'][i]:>12.1f} {baseline_metrics['mean_tpot_ms'][i]:>11.3f} {baseline_metrics['p99_tpot_ms'][i]:>11.3f}")

    print("\n" + "="*90)
    print(f"{disagg_label} 数据 (包含与{baseline_label}的比值):")
    print("="*90)
    print(f"{'Concurrency':>12} {'Mean TTFT':>12} {'P99 TTFT':>12} {'TPS':>11} {'Mean TPOT':>12} {'P99 TPOT':>12} {'TTFT Ratio':>12} {'TPS Ratio':>12} {'TPOT Ratio':>12}")
    print("-"*90)

    for i, x in enumerate(disagg_x):
        # 查找对应的baseline数据
        if x in baseline_x:
            baseline_idx = baseline_x.index(x)
            # 计算比值
            ttft_ratio = calculate_ratio(baseline_metrics['mean_ttft_ms'][baseline_idx],
                                         disagg_metrics['mean_ttft_ms'][i])
            p99_ttft_ratio = calculate_ratio(baseline_metrics['p99_ttft_ms'][baseline_idx],
                                            disagg_metrics['p99_ttft_ms'][i])
            throughput_ratio = calculate_ratio(baseline_metrics['output_throughput'][baseline_idx],
                                              disagg_metrics['output_throughput'][i])
            tpot_ratio = calculate_ratio(baseline_metrics['mean_tpot_ms'][baseline_idx],
                                         disagg_metrics['mean_tpot_ms'][i])
            p99_tpot_ratio = calculate_ratio(baseline_metrics['p99_tpot_ms'][baseline_idx],
                                            disagg_metrics['p99_tpot_ms'][i])

            print(f"{x:>12} {disagg_metrics['mean_ttft_ms'][i]:>11.1f} {disagg_metrics['p99_ttft_ms'][i]:>11.1f} "
                  f"{disagg_metrics['output_throughput'][i]:>11.1f} {disagg_metrics['mean_tpot_ms'][i]:>11.3f} "
                  f"{disagg_metrics['p99_tpot_ms'][i]:>11.3f} {ttft_ratio:>11.2f} {throughput_ratio:>11.2f} {tpot_ratio:>11.2f}")
        else:
            # 如果没有对应的baseline数据，只显示disagg数据
            print(f"{x:>12} {disagg_metrics['mean_ttft_ms'][i]:>11.1f} {disagg_metrics['p99_ttft_ms'][i]:>11.1f} "
                  f"{disagg_metrics['output_throughput'][i]:>11.1f} {disagg_metrics['mean_tpot_ms'][i]:>11.3f} "
                  f"{disagg_metrics['p99_tpot_ms'][i]:>11.3f} {'N/A':>11} {'N/A':>11} {'N/A':>11}")

def main():
    parser = argparse.ArgumentParser(description='生成baseline和disagg配置性能双Y轴对比图')
    parser.add_argument('title', help='graph title')
    parser.add_argument('subtitle', help='graph subtitle')
    parser.add_argument('baseline', help='baseline JSON文件路径')
    parser.add_argument('disagg', help='disagg JSON文件路径')
    parser.add_argument('-o', '--output', default='perf_disagg.png',
                       help='输出图片文件名')

    args = parser.parse_args()

    baseline_data = load_json_data(args.baseline)
    disagg_data = load_json_data(args.disagg)

    baseline_label = extract_config_label(args.baseline)
    disagg_label = extract_config_label(args.disagg)

    create_plot(baseline_data,
                disagg_data,
                baseline_label,
                disagg_label,
                args.title,
                args.subtitle,
                args.output)

if __name__ == "__main__":
    main()
