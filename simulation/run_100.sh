#!/bin/bash

# 循环执行100次
for i in {1..100}
do
    echo "=============================="
    echo "开始第 $i 次执行 (共100次)"
    echo "=============================="
    
    # 执行命令并记录开始时间
    start_time=$(date +%s)
    ./run-copter-sim.sh /home/ame/copter/simulation/mix/acc_webserver_t0.05_l0.7.conf
    
    # 计算命令执行时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "------------------------------"
    echo "命令执行完成，耗时: $duration 秒"
    
    # 如果不是最后一次执行，则等待30秒
    if [ $i -lt 100 ]; then
        echo "等待30秒后开始下一次执行..."
        sleep 30
    else
        echo "已完成所有100次执行"
    fi
done
