#!/usr/bin/env bash
set -euo pipefail

g++ -std=c++17 -shared -fPIC \
    /mnt/d/code/HCCL-Competition/artifacts/dynamic_root_shim.cc \
    -I/home/workspace/hvm/Ascend/cann-9.1.0/x86_64-linux/include \
    -ldl -o /mnt/d/code/HCCL-Competition/artifacts/libdynamic_root_shim.so
