#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#include "struct_header.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


int LB_main(cl_device_id* devicePtr, cl_command_queue* queuePtr, cl_context* contextPtr);
int initialize_data(int_param_struct* intParams, float_param_struct* floatParams);

void analyse_platform(cl_device_id* devices);
void error_check(cl_int err, char* clFunc, bool print);
void read_program_source(char** programSource, const char* programName);
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queue, cl_context* contextPtr);