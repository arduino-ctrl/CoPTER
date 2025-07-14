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
            period = float(group.get("period_s", 1)) * 1e9  # nanoseconds
            # NOTE: `start_time`, `duration`, and `period` are in nanoseconds

            cdf = load_cdf(cdf_path)
            customRand = CustomRand()
            if not customRand.setCdf(cdf):
                print(f"Invalid CDF in {cdf_path}")
                continue

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
                            "id": flow_count,
                            "src": src,
                            "dst": dst,
                            "size": size,
                            "start": t
                        }
                        flow_list.append(flow_data)
                        txt_file.write(f"{src} {dst} 3 100 {size} {t * 1e-9:.9f}\n")
                        flow_count += 1
                        heapq.heapreplace(host_heap, (t + inter_t, src))

            elif pattern == "poisson_incast":
                pass # TODO: Implement incast pattern
            elif pattern == "all_reduce":
                pass # TODO: Implement all_reduce pattern
            elif pattern == "all_to_all":
                pass # TODO: Implement all_to_all pattern
            else:
                print(f"Unknown pattern: {pattern}")
                continue

    with open(output_json, "w") as json_file:
        json.dump(flow_list, json_file, indent=2)

    with open(output_txt, "r+") as txt_file:
        content = txt_file.readlines()
        content[0] = f"{flow_count}\n"
        txt_file.seek(0)
        txt_file.writelines(content)

    print(f"Generated {flow_count} flows. Saved to {output_txt} and {output_json}.")


if __name__ == "__main__":
    main()
