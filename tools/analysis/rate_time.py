import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
from tqdm import tqdm


# ------------------------------
# 数据结构定义
# ------------------------------
@dataclass
class PortMonitor:
    switch_id: int          # 交换机ID
    port_id: int            # 端口ID
    maxrate: int            # 最大速率
    txrate: float           # 发送速率（归一化）
    ecnrate: float          # ECN标记速率（归一化）
    monitor_time_s: float   # 监控时间（秒）


# ------------------------------
# 单文件解析与处理
# ------------------------------
def parse_rate_line(line: str, line_num: int) -> Union[PortMonitor, None]:
    """解析单条速率监控数据，含错误处理"""
    stripped_line = line.strip()
    # 处理空行
    if not stripped_line:
        print(f"警告：第{line_num}行是空行，已跳过")
        return None
    
    parts = stripped_line.split()
    # 检查字段数量（需6个字段：switch_id, port_id, maxrate, txrate, ecnrate, monitor_time_s）
    if len(parts) < 6:
        print(f"警告：第{line_num}行字段不足6个（实际{len(parts)}个），内容：{stripped_line}")
        return None
    if len(parts) > 6:
        print(f"警告：第{line_num}行字段超过6个（实际{len(parts)}个），将使用前6个字段，内容：{stripped_line}")
    
    # 数值类型转换（捕获异常）
    try:
        return PortMonitor(
            switch_id=int(parts[0]),
            port_id=int(parts[1]),
            maxrate=int(parts[2]),
            txrate=float(parts[3]),
            ecnrate=float(parts[4]),
            monitor_time_s=float(parts[5])
        )
    except ValueError as e:
        print(f"警告：第{line_num}行数值转换失败 - {str(e)}，内容：{stripped_line}")
        return None


def process_single_rate_file(
    file_path: str,
    skip_initial_points: int = 2  # 跳过初始不稳定数据点（原代码逻辑保留）
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    """处理单个速率监控文件，返回平均/99分位数的TxRate和EcnRate数组"""
    monitor_records: List[PortMonitor] = []
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    # 读取并解析文件
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            record = parse_rate_line(line, line_num)
            if record:
                monitor_records.append(record)
    
    # 检查有效数据量
    if len(monitor_records) == 0:
        print(f"警告：文件无有效数据 -> {file_path}")
        return None
    if len(monitor_records) <= skip_initial_points:
        print(f"警告：文件数据量不足（{len(monitor_records)}条），无法跳过{skip_initial_points}个初始点 -> {file_path}")
        return None

    # 按时间分桶（同时间戳的数据归为一个桶）
    time_buckets: Dict[float, List[PortMonitor]] = {}
    for record in monitor_records:
        time_key = record.monitor_time_s
        if time_key not in time_buckets:
            time_buckets[time_key] = []
        time_buckets[time_key].append(record)

    # 计算每个时间桶的统计值
    avg_txrate = []
    p99_txrate = []
    avg_ecnrate = []
    p99_ecnrate = []
    
    for time_key, bucket_data in tqdm(time_buckets.items(), desc=f"处理 {os.path.basename(file_path)}"):
        tx_rates = [item.txrate for item in bucket_data]
        ecn_rates = [item.ecnrate for item in bucket_data]
        
        avg_txrate.append((time_key, np.mean(tx_rates)))
        p99_txrate.append((time_key, np.percentile(tx_rates, 99)))
        avg_ecnrate.append((time_key, np.mean(ecn_rates)))
        p99_ecnrate.append((time_key, np.percentile(ecn_rates, 99)))

    # 按时间排序并转换为numpy数组
    avg_txrate.sort(key=lambda x: x[0])
    p99_txrate.sort(key=lambda x: x[0])
    avg_ecnrate.sort(key=lambda x: x[0])
    p99_ecnrate.sort(key=lambda x: x[0])
    
    # 跳过初始不稳定数据点（原代码逻辑）
    avg_txrate_arr = np.array(avg_txrate)[skip_initial_points:]
    p99_txrate_arr = np.array(p99_txrate)[skip_initial_points:]
    avg_ecnrate_arr = np.array(avg_ecnrate)[skip_initial_points:]
    p99_ecnrate_arr = np.array(p99_ecnrate)[skip_initial_points:]

    return avg_txrate_arr, p99_txrate_arr, avg_ecnrate_arr, p99_ecnrate_arr


# ------------------------------
# 绘图功能（多文件对比）
# ------------------------------
def plot_rate_comparison(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None  # 时间窗口聚焦
):
    """绘制多文件的速率对比图（2个子图：平均速率 + 99分位数速率）"""
    plt.figure(figsize=(14, 10))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    line_style_tx = '-'       # TxRate用实线
    line_style_ecn = '--'     # ECNRate用虚线

    # 子图1：平均速率对比（TxRate + ECNRate）
    plt.subplot(2, 1, 1)
    for idx, (filename, (avg_tx, _, avg_ecn, _)) in enumerate(file_results.items()):
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]  # 去掉文件后缀
        
        # 绘制平均TxRate
        plt.plot(
            avg_tx[:, 0], avg_tx[:, 1],
            color=color,
            linestyle=line_style_tx,
            linewidth=2,
            label=f"{file_label} - Avg TxRate"
        )
        
        # 绘制平均ECNRate
        plt.plot(
            avg_ecn[:, 0], avg_ecn[:, 1],
            color=color,
            linestyle=line_style_ecn,
            linewidth=2,
            label=f"{file_label} - Avg ECNRate"
        )
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Rate (normalized)', fontsize=12)
    plt.title(f'{title}\n(Average Rates)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    # 子图2：99分位数速率对比（TxRate + ECNRate）
    plt.subplot(2, 1, 2)
    for idx, (filename, (_, p99_tx, _, p99_ecn)) in enumerate(file_results.items()):
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        # 绘制99分位TxRate
        plt.plot(
            p99_tx[:, 0], p99_tx[:, 1],
            color=color,
            linestyle=line_style_tx,
            linewidth=2,
            label=f"{file_label} - P99 TxRate"
        )
        
        # 绘制99分位ECNRate
        plt.plot(
            p99_ecn[:, 0], p99_ecn[:, 1],
            color=color,
            linestyle=line_style_ecn,
            linewidth=2,
            label=f"{file_label} - P99 ECNRate"
        )
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('99th Percentile Rate (normalized)', fontsize=12)
    plt.title(f'{title}\n(99th Percentile Rates)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 速率对比图保存：{output_path}")


def plot_rate_vs_baseline(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline_filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    """绘制多文件相对于基准文件的速率差异百分比（2个子图：平均速率差 + 99分位数速率差）"""
    # 检查基准文件是否存在
    if baseline_filename not in file_results:
        print(f"警告：基准文件 {baseline_filename} 不在分析结果中，跳过基准对比图")
        return
    
    # 获取基准数据（平均Tx/Ecn、99分位Tx/Ecn）
    baseline_avg_tx, baseline_p99_tx, baseline_avg_ecn, baseline_p99_ecn = file_results[baseline_filename]
    baseline_times = baseline_avg_tx[:, 0]  # 以基准时间轴为统一标准

    plt.figure(figsize=(14, 10))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # 子图1：平均速率差异百分比
    plt.subplot(2, 1, 1)
    for idx, (filename, (avg_tx, _, avg_ecn, _)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue  # 跳过基准文件自身
        
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        # 插值到基准时间轴（确保时间点对齐）
        interp_avg_tx = np.interp(baseline_times, avg_tx[:, 0], avg_tx[:, 1])
        interp_avg_ecn = np.interp(baseline_times, avg_ecn[:, 0], avg_ecn[:, 1])
        
        # 计算差异百分比：(当前值 - 基准值) / 基准值 * 100（避免除以零）
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_diff_pct = (interp_avg_tx - baseline_avg_tx[:, 1]) / baseline_avg_tx[:, 1] * 100
            ecn_diff_pct = (interp_avg_ecn - baseline_avg_ecn[:, 1]) / baseline_avg_ecn[:, 1] * 100
            # 处理基准值为0的特殊情况
            tx_diff_pct[baseline_avg_tx[:, 1] == 0] = 0 if np.all(interp_avg_tx[baseline_avg_tx[:, 1] == 0] == 0) else 100
            ecn_diff_pct[baseline_avg_ecn[:, 1] == 0] = 0 if np.all(interp_avg_ecn[baseline_avg_ecn[:, 1] == 0] == 0) else 100
        
        # 绘制平均TxRate差异
        plt.plot(
            baseline_times, tx_diff_pct,
            color=color,
            linestyle='-',
            linewidth=2,
            label=f"{file_label} - Avg TxRate vs Baseline"
        )
        
        # 绘制平均ECNRate差异
        plt.plot(
            baseline_times, ecn_diff_pct,
            color=color,
            linestyle='--',
            linewidth=2,
            label=f"{file_label} - Avg ECNRate vs Baseline"
        )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # 零差异基准线
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.title(f'{title}\n(Average Rate Difference)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    # 子图2：99分位数速率差异百分比
    plt.subplot(2, 1, 2)
    for idx, (filename, (_, p99_tx, _, p99_ecn)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue
        
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        # 插值到基准时间轴
        interp_p99_tx = np.interp(baseline_times, p99_tx[:, 0], p99_tx[:, 1])
        interp_p99_ecn = np.interp(baseline_times, p99_ecn[:, 0], p99_ecn[:, 1])
        
        # 计算差异百分比
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_diff_pct = (interp_p99_tx - baseline_p99_tx[:, 1]) / baseline_p99_tx[:, 1] * 100
            ecn_diff_pct = (interp_p99_ecn - baseline_p99_ecn[:, 1]) / baseline_p99_ecn[:, 1] * 100
            tx_diff_pct[baseline_p99_tx[:, 1] == 0] = 0 if np.all(interp_p99_tx[baseline_p99_tx[:, 1] == 0] == 0) else 100
            ecn_diff_pct[baseline_p99_ecn[:, 1] == 0] = 0 if np.all(interp_p99_ecn[baseline_p99_ecn[:, 1] == 0] == 0) else 100
        
        # 绘制99分位TxRate差异
        plt.plot(
            baseline_times, tx_diff_pct,
            color=color,
            linestyle='-',
            linewidth=2,
            label=f"{file_label} - P99 TxRate vs Baseline"
        )
        
        # 绘制99分位ECNRate差异
        plt.plot(
            baseline_times, ecn_diff_pct,
            color=color,
            linestyle='--',
            linewidth=2,
            label=f"{file_label} - P99 ECNRate vs Baseline"
        )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.title(f'{title}\n(99th Percentile Rate Difference)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 基准对比图保存：{output_path}")


# ------------------------------
# 批量分析入口
# ------------------------------
def batch_analyze_rate_files(
    file_dir: str,
    file_list: List[str],
    output_dir: str = "rate_analysis_results",
    skip_initial_points: int = 2,
    start_time: float = 2.0,    # 时间窗口起始（秒）
    window_size: float = 0.02,  # 时间窗口长度（20ms）
    baseline_filename: str = None  # 基准文件名（可选）
):
    """批量分析多个速率监控文件，生成对比图和基准差异图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录：{output_dir}")

    # 1. 批量处理所有文件，收集结果
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        result = process_single_rate_file(file_path, skip_initial_points)
        if result:
            file_results[filename] = result
            print(f"✅ 处理完成：{filename}")
        else:
            print(f"❌ 跳过文件：{filename}")

    # 检查是否有有效数据
    if not file_results:
        print("❌ 无有效数据，程序退出")
        return

    # 2. 生成【完整时间跨度对比图】
    full_title = 'Port Rate Comparison (TxRate & ECNRate)'
    full_output = os.path.join(output_dir, "full_time_rate_comparison.png")
    plot_rate_comparison(file_results, full_title, full_output)

    # 3. 生成【指定时间窗口对比图】
    end_time = start_time + window_size
    window_title = f'Port Rate Comparison\n({window_size*1000:.0f}ms Window: {start_time:.3f}-{end_time:.3f}s)'
    window_output = os.path.join(
        output_dir,
        f"window_{start_time:.3f}_{end_time:.3f}_rate_comparison.png"
    )
    plot_rate_comparison(file_results, window_title, window_output, xlim=(start_time, end_time))

    # 4. 若指定基准文件，生成【基准对比图】（完整时间 + 窗口时间）
    if baseline_filename and baseline_filename in file_results:
        # 基准对比图（完整时间）
        baseline_full_title = f'Rate Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]})'
        baseline_full_output = os.path.join(output_dir, "baseline_comparison_full_time.png")
        plot_rate_vs_baseline(file_results, baseline_filename, baseline_full_title, baseline_full_output)

        # 基准对比图（指定时间窗口）
        baseline_window_title = f'Rate Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]} - {window_size*1000:.0f}ms Window)'
        baseline_window_output = os.path.join(
            output_dir,
            f"baseline_comparison_window_{start_time:.3f}_{end_time:.3f}.png"
        )
        plot_rate_vs_baseline(file_results, baseline_filename, baseline_window_title, baseline_window_output, xlim=(start_time, end_time))
    elif baseline_filename:
        print(f"⚠️  基准文件 {baseline_filename} 未在有效文件列表中，跳过基准对比图")


# ------------------------------
# 主函数（配置与启动，直接内置文件参数）
# ------------------------------
if __name__ == "__main__":
    # --------------------------
    # 配置参数：直接在这里修改，无需命令行输入
    # --------------------------
    # 1. 文件路径配置（必须根据实际环境修改！）
    FILE_DIR = "/home/ame/copter/simulation/output"  # 速率文件所在目录
    FILE_LIST = [                                   # 待分析的文件名列表
        "acc_webserver_t0.05_l0.7.txrate",
        "copter_webserver_t0.05_l0.7_m3.txrate",
        "copter_webserver_t0.05_l0.7_like_acc.txrate",
        "copter_webserver_t0.05_l0.7_co.txrate"       # 示例：可作为基准文件
    ]
    
    # 2. 输出配置
    OUTPUT_DIR = "rate_analysis_results"  # 结果输出目录（自动创建）
    
    # 3. 数据处理配置
    SKIP_INITIAL_POINTS = 2               # 跳过初始不稳定数据点数量
    START_TIME = 2.00                     # 时间窗口起始时间（秒）
    WINDOW_SIZE = 0.01                    # 时间窗口长度（秒，0.02即20ms）
    
    # 4. 基准对比配置（可选，需在FILE_LIST中存在）
    BASELINE_FILENAME = "copter_webserver_t0.05_l0.7_co.txrate"

    # # 打印配置信息
    # print("="*50)
    # print("📊 端口速率批量分析配置")
    # print("="*50)
    # print(f"文件目录：{FILE_DIR}")
    # print(f"待分析文件：{FILE_LIST}")
    # print(f"输出目录：{OUTPUT_DIR}")
    # print(f"跳过初始点：{SKIP_INITIAL_POINTS}个")
    # print(f"聚焦窗口：{START_TIME:.3f}s - {START_TIME+WINDOW_SIZE:.3f}s（{WINDOW_SIZE*1000:.0f}ms）")
    # print(f"基准文件：{BASELINE_FILENAME if BASELINE_FILENAME else '未指定'}")
    # print("="*50)

    # 执行批量分析
    batch_analyze_rate_files(
        file_dir=FILE_DIR,
        file_list=FILE_LIST,
        output_dir=OUTPUT_DIR,
        skip_initial_points=SKIP_INITIAL_POINTS,
        start_time=START_TIME,
        window_size=WINDOW_SIZE,
        baseline_filename=BASELINE_FILENAME
    )

    print("\n🎉 批量分析完成！结果已保存至：", OUTPUT_DIR)