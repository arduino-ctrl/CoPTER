import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PortMonitor:
    switch_id: int
    switch_buffer: int
    port_id: int
    port_queue: int
    monitor_time_s: float


def parse_fct_line(line):
    """
    Parse a line from the FCT file. The input format will be like:
    SwitchID SwitchBufferSize PortID PortQueueSize TimeinSeconds
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    switch_id = int(parts[0])
    switch_buffer = int(parts[1])
    port_id = int(parts[2])
    port_queue = int(parts[3])
    monitor_time_s = float(parts[4])
    return PortMonitor(switch_id, switch_buffer, port_id, port_queue, monitor_time_s)


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
    for monitor in monitors:
        bucket_index = monitor.monitor_time_s
        if bucket_index not in buckets:
            buckets[bucket_index] = []
        buckets[bucket_index].append(monitor)

    avg_queue = []
    p99_queue = []

    for bucket_index, bucket_flows in buckets.items():
        if not bucket_flows:
            continue
        avg_queue_size = np.mean([flow.port_queue for flow in bucket_flows])
        p99_queue_size = np.percentile([flow.port_queue for flow in bucket_flows], 99)

        avg_queue.append((bucket_index, avg_queue_size))
        p99_queue.append((bucket_index, p99_queue_size))
    avg_queue.sort(key=lambda x: x[0])
    p99_queue.sort(key=lambda x: x[0])
    avg_queue = np.array(avg_queue)
    p99_queue = np.array(p99_queue)

    plt.figure(figsize=(12, 6))
    plt.plot(avg_queue[:, 0], avg_queue[:, 1], label='Average Queue', color='blue')
    plt.plot(p99_queue[:, 0], p99_queue[:, 1], label='99th Percentile Queue', color='red')
    plt.xlabel('Time (s)') 
    plt.ylabel('Queue (Bytes)')
    plt.title('Average and 99th Percentile Queue over Time')
    plt.legend()
    plt.grid()
    plt.savefig('queue_analysis1.png')
    # plt.show()

    