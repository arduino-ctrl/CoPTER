#!/bin/bash

NS3_DIR=../ns-3.33
NS3_BLD_DIR=$NS3_DIR/build
NS3_LIB_DIR=$(realpath $NS3_BLD_DIR)/lib
LOG_LEV=level_all

LOG_FILE="sim_log_$(date +%Y%m%d_%H%M%S).txt"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <RDMA_CONF>"
    exit 1
fi
RDMA_CONF=$1


if [ ! -f "$RDMA_CONF" ]; then
    echo "Error: Configuration file '$RDMA_CONF' not found"
    exit 1
fi

export LD_LIBRARY_PATH=$NS3_LIB_DIR:$LD_LIBRARY_PATH
export NS_LOG=CongestionControlSimulator=$LOG_LEV
$NS3_BLD_DIR/scratch/copter-sim "$RDMA_CONF"
# export NS_LOG="CongestionControlSimulator=$LOG_LEV:OpenGymInterface=$LOG_LEV"
# $NS3_BLD_DIR/scratch/copter-sim "$RDMA_CONF" > "$LOG_FILE" 2>&1
# echo "log save to:$LOG_FILE"