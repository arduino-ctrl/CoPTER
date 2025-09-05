import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 设置全局字体和样式
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,  # 修复负号显示
    'axes.linewidth': 1.2,  # 坐标轴加粗
    'grid.linestyle': '--',
    'grid.alpha': 0.6,       # 网格透明化
    'figure.dpi': 300,       # 默认分辨率
})

sns.set_style("whitegrid")
sns.set_palette("colorblind")

# -------------------------- 核心改动：自定义子文件夹配置 --------------------------
# 1. 自定义子文件夹名称（用户可根据需求修改，例如按实验场景/日期命名）
custom_folder_name = "webserver_incast"  # 示例：可改为 "test_run"、"flow_size_analysis" 等

# 2. 定义主文件夹路径（固定不变）
main_output_dir = Path("/home/ame/copter/tools/analysis/fct_analysis_plots")

# 3. 拼接最终保存路径（主文件夹 + 自定义子文件夹）
output_dir = main_output_dir / custom_folder_name

# 4. 创建目录（若主文件夹/子文件夹不存在则自动创建，避免报错）
output_dir.mkdir(parents=True, exist_ok=True)
print(f"图表将保存到: {output_dir}")  # 打印最终保存路径，方便用户确认
# --------------------------------------------------------------------------------

# 定义文件路径和方案名称
file_paths = {
    # "copter":"/home/ame/copter/tools/analysis/SECN_result/copter_webserver_t0.05_l0.7_co.fct",
    # "like_acc_copter": "/home/ame/copter/tools/analysis/SECN_result/copter_webserver_t0.05_l0.7_like_acc.fct",
    # "m3":"/home/ame/copter/tools/analysis/SECN_result/copter_webserver_t0.05_l0.7_m3.fct",
    # "acc": "/home/ame/copter/tools/analysis/SECN_result/acc_webserver_t0.05_l0.7.fct",
    # "dcqcn": "/home/ame/copter/tools/analysis/SECN_result/WebServerDCQCN_SECN_load0.7.fct",
    # "hpcc": "/home/ame/copter/tools/analysis/SECN_result/WebServerHPCC_SECN_load0.7.fct"

    # "copter":"/home/ame/copter/tools/analysis/SECN_result/copter_Hadoop_n256_t0.05_l0.9_co.fct",
    # "like_acc_copter": "/home/ame/copter/tools/analysis/SECN_result/copter_Hadoop_n256_t0.05_l0.9_like_acc.fct",
    # "m3":"/home/ame/copter/tools/analysis/SECN_result/copter_Hadoop_n256_t0.05_l0.9_m3.fct",
    # "acc": "/home/ame/copter/tools/analysis/SECN_result/acc_Hadoop_n256_t0.05_l0.9.fct",
    # "dcqcn": "/home/ame/copter/tools/analysis/SECN_result/HadoopDCQCN_SECN_load0.9.fct",
    # "hpcc": "/home/ame/copter/tools/analysis/SECN_result/HadoopHPCC_SECN_load0.9.fct"
    "copter":"/home/ame/copter/tools/analysis/SECN_result/copter_webserver_incast.fct",
    "like_acc_copter": "/home/ame/copter/tools/analysis/SECN_result/copter_webserver_incast_like_acc.fct",
    "m3":"/home/ame/copter/tools/analysis/SECN_result/copter_webserver_incast_m3.fct",
    "acc": "/home/ame/copter/tools/analysis/SECN_result/acc_webserver_incast.fct"
}

color_map = {
    "copter": "#FF6B00",  # 亮橙色 - 高饱和度，视觉冲击力强
    "like_acc_copter": "#0066FF",  # 深蓝色 - 与橙色形成冷暖互补，对比强烈
    "m3": "#E60023",      # 正红色 - 高饱和度，与其他颜色区分度高
    "acc": "#00CC66",     # 亮绿色 - 与红色形成鲜明对比，清新醒目
    "dcqcn": "#9933FF",   # 亮紫色 - 独特色调，与周边颜色无相近色
    "hpcc": "#FFCC00"     # 明黄色 - 高明度，与冷色调形成强烈反差
}

# 折线图样式 - 为每种方案设置独特的线型和标记
line_styles = {
    "copter":'-',
    "like_acc_copter": '-',  # 虚线 - 与copter区分
    "m3": '-',
    "acc": '-',
    "dcqcn": '-',
    "hpcc": '-'
}

markers = {
    "copter": 'x',
    "like_acc_copter": 'o',  # 圆形标记 - 与copter区分
    "m3": 's',
    "acc": '^',
    "dcqcn": 'D',
    "hpcc": 'p'
}

# 柱状图纹理
hatches = {
    "copter": '||',
    "like_acc_copter": 'xx',  # 交叉纹理 - 与copter区分
    "m3": '++',
    "acc": '\\',
    "dcqcn": 'x',
    "hpcc": '+'
}

def parse_fct_file(file_path, debug=False):
    """解析FCT结果文件，提取各类统计数据"""
    data = {
        "overall": None,
        "percent_data": None,
        "size_groups": {}
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if debug:
            print(f"File {file_path} preview:")
            print('\n'.join(lines[:10]) + '\n...')
        
        # 提取整体FCT数据
        overall_pattern = r"Overall FCT:\s+Avg\s+Mid\s+95th\s+99th\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
        overall_match = re.search(overall_pattern, content, re.IGNORECASE)
        
        if not overall_match:
            overall_lines = [line for line in lines if "Overall FCT:" in line]
            if overall_lines:
                overall_nums = []
                for line in overall_lines + [lines[i+1] if i+1 < len(lines) else '' for i, l in enumerate(lines) if "Overall FCT:" in l]:
                    overall_nums.extend(re.findall(r"\d+\.?\d*", line))
                
                if len(overall_nums) >= 4:
                    overall_vals = overall_nums[:4]
                else:
                    raise ValueError(f"Insufficient overall FCT values: {overall_nums}")
            else:
                raise ValueError("No line containing 'Overall FCT:' found")
        else:
            overall_vals = overall_match.groups()
        
        try:
            data["overall"] = {
                "Avg": float(overall_vals[0]),
                "Mid": float(overall_vals[1]),
                "95th": float(overall_vals[2]),
                "99th": float(overall_vals[3])
            }
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error converting overall FCT data: {overall_vals}, error: {e}")
        
        # 提取百分比分组数据
        percent_header = None
        for i, line in enumerate(lines):
            if "Percent" in line and "Size" in line and ("Avg" in line or "Average" in line):
                percent_header = i
                break
        
        if percent_header is None:
            raise ValueError("No percent data header found (line with Percent, Size and Avg)")
        
        # 找到百分比数据的结束位置
        size_group_markers = ["< 100KB", "> 10MB", "< 10KB", "> 1MB"]
        percent_end = None
        for i in range(percent_header + 1, len(lines)):
            if any(marker in lines[i] for marker in size_group_markers):
                percent_end = i
                break
        if percent_end is None:
            percent_end = len(lines)
        
        percent_data = []
        for line in lines[percent_header+1:percent_end]:
            if not line:
                continue
            vals = re.findall(r"\d+\.?\d*", line)
            if len(vals) < 6:
                if debug:
                    print(f"Skipping incomplete percent data line: {line} (found {len(vals)} values)")
                continue
            try:
                percent_data.append({
                    "Percent": float(vals[0]),
                    "Size": float(vals[1]),
                    "Avg": float(vals[2]),
                    "Mid": float(vals[3]),
                    "95th": float(vals[4]),
                    "99th": float(vals[5])
                })
            except (ValueError, IndexError) as e:
                if debug:
                    print(f"Error parsing line: {line}, error: {e}")
        
        if not percent_data:
            raise ValueError("No valid percent data extracted")
        data["percent_data"] = pd.DataFrame(percent_data)
        
        # 提取按大小分组的数据
        size_group_lines = [line for line in lines if any(s in line for s in size_group_markers)]
        for line in size_group_lines:
            group_name = None
            for marker in size_group_markers:
                if marker in line:
                    group_name = marker
                    break
            if not group_name:
                continue
            
            vals = re.findall(r"\d+\.\d+", line)
            if len(vals) >= 4:
                data["size_groups"][group_name] = {
                    "Avg": float(vals[0]),
                    "Mid": float(vals[1]),
                    "95th": float(vals[2]),
                    "99th": float(vals[3])
                }
            else:
                if debug:
                    print(f"Insufficient data for size group {group_name}: {vals}")
        
        missing_groups = [g for g in size_group_markers if g not in data["size_groups"]]
        if missing_groups and debug:
            print(f"Warning: Missing data for size groups: {missing_groups}")
        
        return data
    
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        raise

# 解析所有文件
results = {}
for name, path in file_paths.items():
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Error: File {path} does not exist")
            continue
        
        if not path_obj.is_file():
            print(f"Error: {path} is not a file")
            continue
            
        debug_mode = (name == list(file_paths.keys())[0])
        results[name] = parse_fct_file(path, debug=debug_mode)
        print(f"Successfully parsed {name} data")
    except Exception as e:
        print(f"Error parsing {name}: {e}")

# 设置基准数据
if "copter" in results:
    baseline = results["copter"]
else:
    print("Error: Failed to parse copter data as baseline, exiting")
    exit(1)

# 1. 整体性能对比 - 美化版
def plot_overall_comparison(results):
    metrics = ["Avg", "Mid", "95th", "99th"]
    data = []
    
    for name, res in results.items():
        for metric in metrics:
            data.append({
                "Scheme": name,
                "Metric": metric,
                "Value": res["overall"][metric],
                "Relative Value": res["overall"][metric] / baseline["overall"][metric]
            })
    
    df = pd.DataFrame(data)
    
    # 创建带有子图的图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绝对数值图
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=df, 
                palette=color_map, ax=axes[0], edgecolor='black')
    
    # 为柱状图设置样式：无填充色，加粗边框，保留纹理
    for i, bar in enumerate(axes[0].containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')  # 无填充色
            patch.set_edgecolor(color_map[scheme_name])  # 使用方案对应颜色作为边框色
            patch.set_linewidth(2)  # 加粗边框
            patch.set_hatch(hatches[scheme_name])  # 保留纹理
            patch.set_alpha(1.0)  # 不透明
    
    axes[0].set_title("Overall FCT Performance Comparison", fontsize=16, pad=20)
    axes[0].set_ylabel("FCT slowdown", fontsize=14)
    axes[0].set_xlabel("Performance Metric", fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(title="Scheme", title_fontsize=12, loc='upper left', 
                   bbox_to_anchor=(1, 1))
    
    # 相对数值图 (相对于copter)
    sns.barplot(x="Metric", y="Relative Value", hue="Scheme", data=df, 
                palette=color_map, ax=axes[1], edgecolor='black')
    
    # 为柱状图设置样式：无填充色，加粗边框，保留纹理
    for i, bar in enumerate(axes[1].containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')  # 无填充色
            patch.set_edgecolor(color_map[scheme_name])  # 使用方案对应颜色作为边框色
            patch.set_linewidth(2)  # 加粗边框
            patch.set_hatch(hatches[scheme_name])  # 保留纹理
            patch.set_alpha(1.0)  # 不透明
    
    axes[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='copter baseline')
    axes[1].set_title("Overall FCT Relative to copter", fontsize=16, pad=20)
    axes[1].set_ylabel("Relative Value", fontsize=14)
    axes[1].set_xlabel("Performance Metric", fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(title="Scheme", title_fontsize=12, loc='upper left', 
                   bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_fct_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# 2. 按百分比分组的FCT对比 - 美化版
def plot_percent_comparison(results):
    metrics = ["Avg", "95th", "99th"]
    percent_values = results["copter"]["percent_data"]["Percent"].values
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 18))
    
    for i, metric in enumerate(metrics):
        for name, res in results.items():
            if len(res["percent_data"]) == len(percent_values):
                # 折线图标记去除边框线
                axes[i].plot(percent_values, res["percent_data"][metric], 
                             label=name, color=color_map[name], 
                             linestyle=line_styles[name], marker=markers[name],
                             markersize=6)  # 移除markeredgecolor和markeredgewidth参数
        
        axes[i].set_title(f"{metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"{metric} FCT slowdown", fontsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper left')
        
        # 设置坐标轴范围，留出适当空间
        y_min, y_max = axes[i].get_ylim()
        axes[i].set_ylim(y_min, y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "percent_fct_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 相对值对比图
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 18))
    
    for i, metric in enumerate(metrics):
        for name, res in results.items():
            if name == "copter":
                continue
            if len(res["percent_data"]) == len(baseline["percent_data"]):
                relative_vals = res["percent_data"][metric] / baseline["percent_data"][metric]
                # 折线图标记去除边框线
                axes[i].plot(percent_values, relative_vals, 
                             label=name, color=color_map[name], 
                             linestyle=line_styles[name], marker=markers[name],
                             markersize=6)  # 移除markeredgecolor和markeredgewidth参数
        
        axes[i].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='copter baseline')
        axes[i].set_title(f"Relative {metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"Relative {metric} FCT", fontsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper left')
        
        # 设置坐标轴范围，留出适当空间
        y_min, y_max = axes[i].get_ylim()
        axes[i].set_ylim(max(0, y_min * 0.95), y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "percent_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# 3. 按流大小分组的对比 - 美化版
def plot_size_group_comparison(results):
    size_groups = ["< 10KB", "< 100KB", "> 1MB", "> 10MB"]
    metrics = ["Avg", "Mid", "95th", "99th"]
    
    data = []
    for group in size_groups:
        for name, res in results.items():
            if group not in res["size_groups"]:
                print(f"Warning: No data for {group} in {name}")
                continue
            for metric in metrics:
                data.append({
                    "Flow Size Group": group,
                    "Scheme": name,
                    "Metric": metric,
                    "Value": res["size_groups"][group][metric],
                    "Relative Value": res["size_groups"][group][metric] / baseline["size_groups"][group][metric]
                })
    
    df = pd.DataFrame(data)
    if df.empty:
        print("Error: Insufficient size group data for plotting")
        return
    
    # 绝对数值图
    g = sns.catplot(x="Metric", y="Value", hue="Scheme", col="Flow Size Group",
                    data=df, kind="bar", palette=color_map, col_wrap=2, height=5,
                    edgecolor='black')
    
    # 为每个子图的柱状图设置样式：无填充色，加粗边框，保留纹理
    for i, ax in enumerate(g.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j]
            for patch in bar.patches:
                patch.set_facecolor('none')  # 无填充色
                patch.set_edgecolor(color_map[scheme_name])  # 使用方案对应颜色作为边框色
                patch.set_linewidth(2)  # 加粗边框
                patch.set_hatch(hatches[scheme_name])  # 保留纹理
                patch.set_alpha(1.0)  # 不透明
    
    g.set_titles("{col_name} Flows", fontsize=14)
    g.set_axis_labels("Performance Metric", "FCT slowdown")
    g.legend.set_title("Scheme")
    g.legend.set_bbox_to_anchor((1.05, 1))
    
    # 添加网格线
    for ax in g.axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 相对值对比图
    rel_df = df[df["Scheme"] != "copter"]
    g = sns.catplot(x="Metric", y="Relative Value", hue="Scheme", col="Flow Size Group",
                    data=rel_df, kind="bar", palette=color_map, col_wrap=2, height=5,
                    edgecolor='black')
    
    # 为每个子图的柱状图设置样式：无填充色，加粗边框，保留纹理
    for i, ax in enumerate(g.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j+1]  # +1 because we excluded copter
            for patch in bar.patches:
                patch.set_facecolor('none')  # 无填充色
                patch.set_edgecolor(color_map[scheme_name])  # 使用方案对应颜色作为边框色
                patch.set_linewidth(2)  # 加粗边框
                patch.set_hatch(hatches[scheme_name])  # 保留纹理
                patch.set_alpha(1.0)  # 不透明
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
    
    g.set_titles("{col_name} Flows", fontsize=14)
    g.set_axis_labels("Performance Metric", "Relative FCT")
    g.legend.set_title("Scheme")
    g.legend.set_bbox_to_anchor((1.05, 1))
    
    # 添加网格线
    for ax in g.axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# 执行所有可视化
if __name__ == "__main__":
    if len(results) < 2:
        print("Error: Insufficient data parsed, need at least 2 schemes for comparison")
        exit(1)
    
    print("Generating overall performance comparison plot...")
    plot_overall_comparison(results)
    
    print("Generating percentile group comparison plots...")
    plot_percent_comparison(results)
    
    print("Generating flow size group comparison plots...")
    # plot_size_group_comparison(results)
    
    print(f"All analysis plots have been generated and saved to: {output_dir}")