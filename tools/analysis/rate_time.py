import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class PortMonitor:
    switch_id: int
    port_id: int
    maxrate: int
    txrate: float
    ecnrate: float
    monitor_time_s: float

def parse_fct_line(line):
    """
    Parse a line from the FCT file. The input format will be like:
    SwitchID PortID TxRate EcnRate TimeinSeconds
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    switch_id = int(parts[0])
    port_id = int(parts[1])
    maxrate = int(parts[2])
    txrate = float(parts[3])
    ecnrate = float(parts[4])
    monitor_time_s = float(parts[5])
    return PortMonitor(switch_id, port_id, maxrate, txrate, ecnrate, monitor_time_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', dest='filepath', action='store', default=None, help="Specify the path of the fct files")
    args = parser.parse_args()

    if args.filepath is None or not os.path.exists(args.filepath):
        print("Please specify a valid file path.")
        exit(1)
    
    filepath = args.filepath

    monitors = []
    with open(filepath, 'r') as f:
        for line in f:
            monitor = parse_fct_line(line)
            if monitor is not None:
                monitors.append(monitor)
    
    # Calculate average FCT and 99th percentile FCT in buckets, the bucket size is 100us = 100000 ns
    buckets = {}
    for monitor in tqdm(monitors):
        bucket_index = monitor.monitor_time_s
        if bucket_index not in buckets:
            buckets[bucket_index] = []
        buckets[bucket_index].append(monitor)

    avg_txrate = []
    p99_txrate = []
    avg_ecnrate = []
    p99_ecnrate = []

    for bucket_index, bucket_flows in tqdm(buckets.items()):
        if not bucket_flows:
            continue
        avg_txrate_bucket = np.mean([flow.txrate for flow in bucket_flows])
        p99_txrate_bucket = np.percentile([flow.txrate for flow in bucket_flows], 99)
        avg_ecnrate_bucket = np.mean([flow.ecnrate for flow in bucket_flows])
        p99_ecnrate_bucket = np.percentile([flow.ecnrate for flow in bucket_flows], 99)

        avg_txrate.append((bucket_index, avg_txrate_bucket))
        p99_txrate.append((bucket_index, p99_txrate_bucket))
        avg_ecnrate.append((bucket_index, avg_ecnrate_bucket))
        p99_ecnrate.append((bucket_index, p99_ecnrate_bucket))
    
    avg_txrate.sort(key=lambda x: x[0])
    p99_txrate.sort(key=lambda x: x[0])
    avg_ecnrate.sort(key=lambda x: x[0])
    p99_ecnrate.sort(key=lambda x: x[0])
    avg_txrate = np.array(avg_txrate)
    p99_txrate = np.array(p99_txrate)
    avg_ecnrate = np.array(avg_ecnrate)
    p99_ecnrate = np.array(p99_ecnrate)

    plt.subplot(2, 1, 1)
    plt.plot(avg_txrate[2:, 0], avg_txrate[2:, 1], label='Average Tx Rate', color='blue')
    plt.plot(avg_ecnrate[2:, 0], avg_ecnrate[2:, 1], label='Average ECN Rate', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Rate (normalized)')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(p99_txrate[2:, 0], p99_txrate[2:, 1], label='99th Percentile Tx Rate', color='red')
    plt.plot(p99_ecnrate[2:, 0], p99_ecnrate[2:, 1], label='99th Percentile ECN Rate', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('99th Percentile Rate (normalized)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('rate_analysis.png')
    # plt.show()  # Uncomment to display the plot interactively