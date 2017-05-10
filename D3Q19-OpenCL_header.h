#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define TYPE_INT 0
#define TYPE_FLOAT 1
#define TYPE_INT_3VEC 2
#define TYPE_FLOAT_3VEC 3
#define TYPE_STRING 4

#define BC_PERIODIC 0
#define BC_BOUNCE_BACK 1
#define BC_VELOCITY 2

#define LB_Q 19

#include "struct_header.h"

int LB_main(cl_device_id* devicePtr, cl_command_queue* queuePtr, cl_context* contextPtr);
	
int initialize_data(int_param_struct* intParams, float_param_struct* floatParams,
	host_param_struct* hostParams);
int initialize_distribution(host_param_struct* hostParams, int_param_struct* intParams,
	cl_float* fHost);
int equilibrium_distribution_D3Q19(float rho, float* vel, float* f_eq);
	
int create_stream_mapping(int_param_struct* intDat);
int process_input_line(char* fLine, input_data_struct* inputDefaults, int inputDefaultSize);
int display_input_params(int_param_struct* intParams, float_param_struct* floatParams);

int create_LB_kernels(cl_context* contextPtr, cl_device_id* devices, cl_kernel* kernelCPU, 
	cl_kernel* kernelGPU);
int print_program_build_log(cl_program* program, cl_device_id* device);

void analyse_platform(cl_device_id* devices);
int error_check(cl_int err, char* clFunc, bool print);
void read_program_source(char** programSource, const char* programName);
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queue, cl_context* contextPtr);