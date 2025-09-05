# TraGen.py - Traffic Generator for Network Simulation
# The name `TraGen` is motivated by Dr. CG Lin's `TraGe` paper.
# This script generates network traffic based on specified flow groups (different hosts, start time, and durations) and configurations.
# It supports different traffic patterns such as Poisson, All-Reduce, and All-to-All.

import sys
import os
import random
import math
import heapq
import json
from optparse import OptionParser


class CustomRand:
    def __init__(self):
        pass

    def testCdf(self, cdf):
        if cdf[0][1] != 0 or cdf[-1][1] != 100:
            return False
        for i in range(1, len(cdf)):
            if cdf[i][1] <= cdf[i - 1][1] or cdf[i][0] <= cdf[i - 1][0]:
                return False
        return True

    def setCdf(self, cdf):
        if not self.testCdf(cdf):
            return False
        self.cdf = cdf
        return True

    def getAvg(self):
        s = 0
        last_x, last_y = self.cdf[0]
        for x, y in self.cdf[1:]:
            s += (x + last_x) / 2.0 * (y - last_y)
            last_x, last_y = x, y
        return s / 100

    def rand(self):
        r = random.random() * 100
        return self.getValueFromPercentile(r)

    def getValueFromPercentile(self, y):
        for i in range(1, len(self.cdf)):
            if y <= self.cdf[i][1]:
                x0, y0 = self.cdf[i - 1]
                x1, y1 = self.cdf[i]
                return x0 + (x1 - x0) / (y1 - y0) * (y - y0)


def translate_bandwidth(b):
    if not b or not isinstance(b, str):
        return None
    if b.endswith('G'):
        return float(b[:-1]) * 1e9
    if b.endswith('M'):
        return float(b[:-1]) * 1e6
    if b.endswith('K'):
        return float(b[:-1]) * 1e3
    return float(b)


def poisson(lam):
    return -math.log(1 - random.random()) * lam


def load_cdf(file_path):
    with open(file_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]


def main():
    parser = OptionParser()
    parser.add_option("-b", "--bandwidth", dest="bandwidth", default="10G",
                      help="bandwidth of host link (G/M/K), default 10G")
    parser.add_option("-c", "--config", dest="group_config", default=None,
                      help="JSON file specifying flow generation groups")
    options, _ = parser.parse_args()

    if not options.group_config:
        print("Usage: --flow-groups <group_file.json> required")
        sys.exit(1)

    bandwidth = translate_bandwidth(options.bandwidth)
    config_dir = os.path.dirname(options.group_config)
    config_name = os.path.basename(options.group_config)[:-7]

    # FIXME: Output path is not the same as the input path `\pattern`
    output_txt = os.path.join("result", config_name+".flow")
    output_json = os.path.join("result", config_name+"_flows.json")

    if bandwidth is None:
        print("Bandwidth format incorrect")
        sys.exit(1)

    with open(options.group_config, 'r') as f:
        group_config = json.load(f)

    flow_list = []
    flow_count = 0

    with open(output_txt, "w") as txt_file:
        txt_file.write("0\n")  # placeholder

        for group in group_config:
            src_hosts = group["src_hosts"]
            dst_hosts = group["dst_hosts"]
            cdf_path = os.path.join(config_dir, group["cdf"])
            start_time = int(group.get("start_time_s", 2) * 1e9)  # nanoseconds
            duration = int(group.get("duration_s", 10) * 1e9)  # nanoseconds
            load = float(group.get("load", 0.3))
            pattern = group.get("pattern", "poisson")
            period = float(group.get("period_s", 1)) * 1e9  # nanoseconds，周期性模式的周期（纳秒）
            incast_dst_count = int(group.get("incast_dst_count", 1))  # Incast汇聚目的数（默认1）
            reduce_group_size = int(group.get("reduce_group_size", 8)) # All-Reduce组大小（默认8，贴合Rack内主机数）
            # NOTE: `start_time`, `duration`, and `period` are in nanoseconds

            cdf = load_cdf(cdf_path)
            customRand = CustomRand()
            if not customRand.setCdf(cdf):
                print(f"Invalid CDF in {cdf_path}")
                continue
            avg_size = customRand.getAvg()  # 流大小的平均值（字节）

            if pattern == "poisson_random":
                avg_size = customRand.getAvg()
                avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

                host_heap = [(start_time + int(poisson(avg_inter_arrival)), h) for h in src_hosts]
                heapq.heapify(host_heap)  # Min-heap for host events

                while host_heap:
                    t, src = host_heap[0]
                    inter_t = int(poisson(avg_inter_arrival))
                    dst = random.choice(dst_hosts)
                    while dst == src:
                        dst = random.choice(dst_hosts)

                    if t + inter_t > start_time + duration:
                        heapq.heappop(host_heap)
                    else:
                        size = max(1, int(customRand.rand()))
                        flow_data = {
                            "id": int(flow_count),
                            "src": int(src),
                            "dst": int(dst),
                            "size": int(size),
                            "start": int(flow_start)  # 强制int
                        }
                        flow_list.append(flow_data)
                        txt_file.write(f"{src} {dst} 3 100 {size} {t * 1e-9:.9f}\n")
                        flow_count += 1
                        heapq.heapreplace(host_heap, (t + inter_t, src))

            ###########################################################################
            # 2. 泊松入向汇聚模式（poisson_incast）：多源→少目的，泊松到达
            ###########################################################################
            elif pattern == "poisson_incast":
                # 校验：汇聚目的数不能超过目的主机总数
                if incast_dst_count > len(dst_hosts):
                    print(f"incast_dst_count ({incast_dst_count}) > len(dst_hosts) ({len(dst_hosts)})")
                    continue
                # 选择固定的汇聚目的主机（如Rack内的1台主节点）
                incast_dsts = random.sample(dst_hosts, k=incast_dst_count)
                print(f"Poisson Incast: {len(src_hosts)} sources → {incast_dst_count} destinations ({incast_dsts})")

                # 计算平均到达间隔（与poisson_random一致，基于负载控制）
                avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

                # 初始化小根堆：每个源主机的第一个流时间
                host_heap = [(start_time + int(poisson(avg_inter_arrival)), h) for h in src_hosts]
                heapq.heapify(host_heap)

                while host_heap:
                    t, src = host_heap[0]
                    inter_t = int(poisson(avg_inter_arrival))
                    # 从固定汇聚目的中选一个（避免源=目的）
                    dst = random.choice(incast_dsts)
                    while dst == src:
                        dst = random.choice(incast_dsts)

                    # 检查时间窗口
                    if t + inter_t > start_time + duration:
                        heapq.heappop(host_heap)
                    else:
                        size = max(1, int(customRand.rand()))
                        flow_data = {
                            "id": int(flow_count),
                            "src": int(src),
                            "dst": int(dst),
                            "size": int(size),
                            "start": int(flow_start)  # 强制int
                        }
                        flow_list.append(flow_data)
                        txt_file.write(f"{src} {dst} 3 100 {size} {t * 1e-9:.9f}\n")
                        flow_count += 1
                        heapq.heapreplace(host_heap, (t + inter_t, src))
            ###########################################################################
            # 3. 全归约模式（all_reduce）：组内协同聚合，周期性同步
            ###########################################################################
            elif pattern == "all_reduce":
                # 划分All-Reduce组（贴合拓扑：按Rack划分，每组8台主机）
                reduce_groups = []
                # 从src_hosts中按reduce_group_size拆分（确保组内无重复）
                for i in range(0, len(src_hosts), reduce_group_size):
                    group = src_hosts[i:i+reduce_group_size]
                    if len(group) >= 2:  # 组内至少2台主机才有效
                        reduce_groups.append(group)
                if not reduce_groups:
                    print("No valid All-Reduce groups (need at least 2 hosts per group)")
                    continue
                print(f"All-Reduce: {len(reduce_groups)} groups, each with {reduce_group_size} hosts")

                # 计算周期数（在时间窗口内可触发多少次All-Reduce）
                max_cycle = int(duration // period)
                for cycle in range(max_cycle):
                    # 每个周期的起始时间（加微小偏移避免所有流完全同步）
                    cycle_start = start_time + cycle * period
                    if cycle_start > start_time + duration:
                        break

                    # 遍历每个All-Reduce组，生成组内所有非自环流
                    for reduce_group in reduce_groups:
                        for src in reduce_group:
                            for dst in reduce_group:
                                if src == dst:
                                    continue  # 跳过自环

                                # 流启动时间：周期起始时间 + 微秒级偏移（模拟同步误差）
                                time_offset = random.randint(0, 1000)  # 0-1微秒偏移
                                flow_start = cycle_start + time_offset
                                # 采样流大小
                                size = max(1, int(customRand.rand()))
                                flow_data = {
                                    "id": int(flow_count),
                                    "src": int(src),
                                    "dst": int(dst),
                                    "size": int(size),
                                    "start": int(flow_start)  # 强制int
                                }
                                flow_list.append(flow_data)
                                txt_file.write(f"{src} {dst} 3 100 {size} {flow_start * 1e-9:.9f}\n")
                                flow_count += 1
            ###########################################################################
            # 4. 全对全模式（all_to_all）：每个源→每个目的，泊松到达
            ###########################################################################
            elif pattern == "all_to_all":
                 # 校验：源和目的集合非空，且避免自环
                valid_src = [s for s in src_hosts if s not in dst_hosts]
                if not valid_src:
                    valid_src = src_hosts  # 允许源在目的集合中（但会跳过自环）
                if not src_hosts or not dst_hosts:
                    print("src_hosts or dst_hosts is empty for all_to_all")
                    continue
                print(f"All-to-All: {len(src_hosts)} sources → {len(dst_hosts)} destinations")

                # 计算平均到达间隔（基于负载控制）
                avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

                # 初始化小根堆：每个源+当前要发送的目的索引（确保遍历所有目的）
                # 堆元素：(下一个流时间, 源主机, 目的索引)
                host_heap = []
                for src in src_hosts:
                    first_time = start_time + int(poisson(avg_inter_arrival))
                    host_heap.append((first_time, src, 0))
                heapq.heapify(host_heap)

                while host_heap:
                    t, src, dst_idx = host_heap[0]
                    inter_t = int(poisson(avg_inter_arrival))
                    # 按索引选择目的（循环遍历所有目的）
                    dst = dst_hosts[dst_idx % len(dst_hosts)]
                    # 跳过自环：更新目的索引直到找到非自环目的
                    while dst == src:
                        dst_idx += 1
                        dst = dst_hosts[dst_idx % len(dst_hosts)]

                    # 检查时间窗口
                    if t + inter_t > start_time + duration:
                        heapq.heappop(host_heap)
                    else:
                        size = max(1, int(customRand.rand()))
                        flow_data = {
                            "id": int(flow_count),
                            "src": int(src),
                            "dst": int(dst),
                            "size": int(size),
                            "start": int(flow_start)  # 强制int
                        }
                        flow_list.append(flow_data)
                        txt_file.write(f"{src} {dst} 3 100 {size} {t * 1e-9:.9f}\n")
                        flow_count += 1
                        # 更新堆：下一个流时间 + 目的索引+1（遍历下一个目的）
                        next_dst_idx = dst_idx + 1
                        heapq.heapreplace(host_heap, (t + inter_t, src, next_dst_idx))
            else:
                print(f"Unknown pattern: {pattern}")
                continue

    with open(output_json, "w") as json_file:
        json.dump(
            flow_list, 
            json_file, 
            indent=2,
            default=lambda x: int(x) if isinstance(x, float) and x.is_integer() else x  # 浮点数转整数（如200.0→200）
        )

    with open(output_txt, "r+") as txt_file:
        content = txt_file.readlines()
        content[0] = f"{flow_count}\n"
        txt_file.seek(0)
        txt_file.writelines(content)

    print(f"Generated {flow_count} flows. Saved to {output_txt} and {output_json}.")


if __name__ == "__main__":
    main()
