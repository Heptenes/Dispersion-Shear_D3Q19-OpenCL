#!/bin/bash # 

cp struct_header_host.h struct_header_device.h

sed -i '' -e 's/cl_int/int/g' struct_header_device.h
sed -i '' -e 's/cl_float/float/g' struct_header_device.h

gcc D3Q19-OpenCL.c -o D3Q19-OpenCL_bin.out -framework OpenCL -Wall
