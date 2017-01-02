#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void analyse_platform(cl_device_id* devices);
void error_check(cl_int err, char* clFunc, bool print);
void read_program_source(char** programSource, const char* programName);
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queue, cl_context* contextPtr);