import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
import time

@dataclass
class PortQueueData:
    switch_id: int          
    switch_buffer: int      
    port_id: int            
    queue_size: int         
    monitor_time_s: float   


def parse_queue_line(line: str, line_num: int) -> Union[PortQueueData, None]:
    stripped_line = line.strip()
    if not stripped_line:
        print(f"警告：第{line_num}行是空行，已跳过")
        return None
        
    parts = stripped_line.split()
    if len(parts) < 5:
        print(f"警告：第{line_num}行字段不足5个（实际{len(parts)}个），内容：{stripped_line}")
        return None
    if len(parts) > 5:
        print(f"警告：第{line_num}行字段超过5个（实际{len(parts)}个），将使用前5个字段，内容：{stripped_line}")
    
    try:
        return PortQueueData(
            switch_id=int(parts[0]),
            switch_buffer=int(parts[1]),
            port_id=int(parts[2]),
            queue_size=int(parts[3]),
            monitor_time_s=float(parts[4])
        )
    except ValueError as e:
        print(f"警告：第{line_num}行数值转换失败 - {str(e)}，内容：{stripped_line}")
        return None


def process_single_queue_file(file_path: str) -> Union[Tuple[np.ndarray, np.ndarray], None]:
    queue_records: List[PortQueueData] = []
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  
        for line_num, line in enumerate(f, 1):  
            record = parse_queue_line(line, line_num)
            if record:
                queue_records.append(record)
    
    if len(queue_records) == 0:
        print(f"警告：文件无有效数据 -> {file_path}")
        return None

    time_buckets = {}
    for record in queue_records:
        time_key = record.monitor_time_s
        if time_key not in time_buckets:
            time_buckets[time_key] = []
        time_buckets[time_key].append(record)

    avg_queue = []
    p99_queue = []
    for time_key, bucket_data in time_buckets.items():
        queue_sizes = [item.queue_size for item in bucket_data]
        avg_size = np.mean(queue_sizes)
        p99_size = np.percentile(queue_sizes, 99)
        avg_queue.append((time_key, avg_size))
        p99_queue.append((time_key, p99_size))

    avg_queue.sort(key=lambda x: x[0])
    p99_queue.sort(key=lambda x: x[0])

    return np.array(avg_queue), np.array(p99_queue)


def plot_queue(
    file_results: dict,  # {filename: (avg_array, p99_array)}
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None  # 控制X轴范围，实现“聚焦”
):
    plt.figure(figsize=(14, 8))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    line_style_avg = '-'
    line_style_p99 = '--'

    for idx, (filename, (avg_array, p99_array)) in enumerate(file_results.items()):
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        # 绘制平均队列
        plt.plot(
            avg_array[:, 0], avg_array[:, 1],
            color=color,
            linestyle=line_style_avg,
            linewidth=2,
            label=f"{file_label} - Avg Queue"
        )
        
        # 绘制99th Percentile队列（如需启用，取消注释）
        # plt.plot(
        #     p99_array[:, 0], p99_array[:, 1],
        #     color=color,
        #     linestyle=line_style_p99,
        #     linewidth=2,
        #     label=f"{file_label} - 99th Percentile Queue"
        # )

    if xlim:
        plt.xlim(*xlim)  # 聚焦时间窗口
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Queue Size (Bytes)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(
        fontsize=10, 
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表保存：{output_path}")


def plot_comparison_against_baseline(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    baseline_filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    """
    绘制各文件相对于基准文件的队列大小差异百分比
    """
    if baseline_filename not in file_results:
        print(f"警告：基准文件 {baseline_filename} 不在分析结果中，无法生成对比图")
        return
    
    # 获取基准数据
    baseline_avg, _ = file_results[baseline_filename]
    baseline_times = baseline_avg[:, 0]
    baseline_values = baseline_avg[:, 1]
    
    plt.figure(figsize=(14, 8))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    
    # 为每个文件计算与基准的差异
    for idx, (filename, (avg_array, _)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue  # 跳过基准文件本身
            
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        # 插值以匹配基准时间点，确保可以比较
        interp_values = np.interp(baseline_times, avg_array[:, 0], avg_array[:, 1])
        
        # 计算差异百分比 (当前值 - 基准值) / 基准值 * 100
        # 避免除以零的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            diff_percent = (interp_values - baseline_values) / baseline_values * 100
            # 处理基准值为0的情况
            diff_percent[baseline_values == 0] = 0 if np.all(interp_values[baseline_values == 0] == 0) else 100
        
        plt.plot(
            baseline_times, diff_percent,
            color=color,
            linewidth=2,
            label=f"{file_label} vs Baseline (%)"
        )
    
    # 添加零基准线
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    if xlim:
        plt.xlim(*xlim)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(
        fontsize=10, 
        loc='upper left',
        bbox_to_anchor=(1, 1)
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 基准对比图表保存：{output_path}")


def batch_analyze_queue_files(
    file_dir: str, 
    file_list: List[str], 
    main_output_dir: str = "queue_analysis_results",  # 主输出目录（固定）
    custom_subfolder: str = "webserver_load0.7_202405",  # 自定义子文件夹（用户可改）
    start_time: float = 2.00,  # 时间窗口起始（秒）
    window_size: float = 0.02,  # 时间窗口长度（20ms = 0.02秒）
    baseline_filename: str = "copter_webserver_t0.05_l0.7_co.queue"  # 基准文件名
):
    # -------------------------- 核心改动：拼接主目录+自定义子文件夹 --------------------------
    # 最终输出路径：主输出目录 / 自定义子文件夹
    final_output_dir = os.path.join(main_output_dir, custom_subfolder)
    # 自动创建目录（包括主目录和子文件夹，不存在则创建，已存在则跳过）
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📁 所有图表将保存到：{final_output_dir}")
    # --------------------------------------------------------------------------------

    # 1. 处理所有文件，收集数据
    file_results = {}
    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        result = process_single_queue_file(file_path)
        if result:
            file_results[filename] = result
            print(f"✅ 处理完成：{filename}")
        else:
            print(f"❌ 跳过文件：{filename}")

    if not file_results:
        print("❌ 无有效数据，程序退出")
        return

    # 2. 生成【完整时间跨度图】（保存到自定义子文件夹）
    full_title = 'Port Queue Size Comparison\n(Average & 99th Percentile - Full Time)'
    full_output = os.path.join(final_output_dir, "full_time_queue_comparison.png")  # 固定文件名
    plot_queue(file_results, full_title, full_output)

    # 3. 生成【指定时间窗口图】（保存到自定义子文件夹）
    end_time = start_time + window_size
    window_title = f'Port Queue Size Comparison\n({window_size*1000:.0f}ms Window: {start_time:.3f}-{end_time:.3f}s)'
    window_output = os.path.join(
        final_output_dir, 
        f"window_{start_time:.3f}_{end_time:.3f}_queue_comparison.png"  # 含时间标识
    )
    plot_queue(file_results, window_title, window_output, xlim=(start_time, end_time))
    
    # 4. 生成【基准对比图】- 完整时间（保存到自定义子文件夹）
    baseline_full_title = f'Queue Size Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]})'
    baseline_full_output = os.path.join(final_output_dir, "baseline_comparison_full_time.png")
    plot_comparison_against_baseline(file_results, baseline_filename, baseline_full_title, baseline_full_output)
    
    # 5. 生成【基准对比图】- 指定时间窗口（保存到自定义子文件夹）
    baseline_window_title = f'Queue Size Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]} - {window_size*1000:.0f}ms Window)'
    baseline_window_output = os.path.join(
        final_output_dir, 
        f"baseline_comparison_window_{start_time:.3f}_{end_time:.3f}.png"
    )
    plot_comparison_against_baseline(file_results, baseline_filename, baseline_window_title, baseline_window_output, xlim=(start_time, end_time))


if __name__ == "__main__":
    # -------------------------- 配置参数（用户可根据需求修改） --------------------------
    QUEUE_FILE_DIR = "/home/ame/copter/simulation/output"  # 队列数据文件所在目录
    QUEUE_FILE_LIST = [  # 需要分析的队列数据文件
        # "acc_webserver_t0.05_l0.7.queue",
        # "copter_webserver_t0.05_l0.7_m3.queue",
        # "copter_webserver_t0.05_l0.7_like_acc.queue",
        # "copter_webserver_t0.05_l0.7_co.queue"  # 基准文件

        # "acc_Hadoop_n256_t0.05_l0.9.queue",
        # "copter_Hadoop_n256_t0.05_l0.9_m3.queue",
        # "copter_Hadoop_n256_t0.05_l0.9_like_acc.queue",
        # "copter_Hadoop_n256_t0.05_l0.9_co.queue"

        "acc_webserver_incast.queue",
        "copter_webserver_incast.queue"
    ]
    MAIN_OUTPUT_DIR = "queue_analysis_results"  # 主输出目录（固定，无需频繁修改）
    CUSTOM_SUBFOLDER = "webserver_incast"  # 自定义子文件夹（核心！按场景命名）
    START_TIME = 2.03   # 时间窗口起始（秒，根据数据调整）
    WINDOW_SIZE = 0.02  # 时间窗口长度（20ms = 0.02秒）
    BASELINE_FILENAME = "copter_webserver_incast.queue"  # 基准文件名
    # --------------------------------------------------------------------------------

    # 执行批量分析（传入自定义子文件夹参数）
    batch_analyze_queue_files(
        file_dir=QUEUE_FILE_DIR,
        file_list=QUEUE_FILE_LIST,
        main_output_dir=MAIN_OUTPUT_DIR,
        custom_subfolder=CUSTOM_SUBFOLDER,  # 关键：将图表保存到自定义子文件夹
        start_time=START_TIME,
        window_size=WINDOW_SIZE,
        baseline_filename=BASELINE_FILENAME
    )