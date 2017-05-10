#!/bin/bash # 

cp struct_header.h struct_header_CL.h

sed -i '' -e 's/cl_int/int/g' struct_header_CL.h
sed -i '' -e 's/cl_float/float/g' struct_header_CL.h

gcc D3Q19-OpenCL_main.c -o D3Q19-OpenCL_bin.out -framework OpenCL
