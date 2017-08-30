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
#define TYPE_FLOAT_4VEC 4
#define TYPE_STRING 5

#define BC_PERIODIC 0
#define BC_BOUNCE_BACK 1
#define BC_VELOCITY 2

#define WORD_STRING_SIZE 64

#define LB_Q 19

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// X-macros
#define LIST_OF_KERNELS \
    X(GPU_collideSRT_stream_D3Q19) \
	X(GPU_boundary_velocity) \
	X(GPU_boundary_periodic) \
	X(CPU_sphere_collide)

#include "struct_header_host.h"

// Structs not used by kernel programs declared here
typedef struct {

	int ConsolePrintFreq;
	char InitialDist[WORD_STRING_SIZE];
	float InitialVel[3];
    int InitialParticleDistribution;
    float InitialParticleBuffer;
    int InterpOrderIBM;

} host_param_struct;

typedef struct {
#define X(kernelName) cl_kernel kernelName;
	LIST_OF_KERNELS
#undef X
} kernel_struct;

typedef struct {
	cl_command_queue kernel_queue;
	cl_device_id kernel_device;
} kernel_data_struct;

typedef struct {

	char keyword[WORD_STRING_SIZE];
	int dataType;
	void* varPtr;
	char defString[WORD_STRING_SIZE];

} input_data_struct;


int simulation_main(cl_device_id* devices, cl_command_queue* queueCPU, cl_command_queue* queueGPU,
	cl_context* contextPtr);
void particle_dynamics(int_param_struct* intDat, cl_float4* parKinematics_h, cl_float4* parForces_h);

int initialize_data(int_param_struct* intParams, flp_param_struct* floatParams, host_param_struct* hostParams);
int parameter_checking(int_param_struct* intDat, flp_param_struct* flpDat);

void initialize_lattice_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
    cl_float* f_h, cl_float* g_h, cl_float* u_h);
void initialize_particle_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float4* parKinematics_h, cl_float4* parForces_h, cl_uint* pointIDs_h);
int equilibrium_distribution_D3Q19(float rho, float* vel, float* f_eq);

int process_input_line(char* fLine, input_data_struct* inputDefaults, int inputDefaultSize);
void sphere_discretization(int_param_struct* intDat, flp_param_struct* flpDat, cl_float4* sphereNodesXYZ);

int create_periodic_stream_mapping(int_param_struct* intDat, cl_int** strMapPtr);
int write_lattice_field(cl_float* u_h, int_param_struct* intDat);

int create_LB_kernels(cl_context* contextPtr, cl_device_id* devices, kernel_struct* kernelDat);

int display_input_params(int_param_struct* intParams, flp_param_struct* floatParams);
int print_program_build_log(cl_program* program, cl_device_id* device);

void analyse_platform(cl_device_id* devices);
int error_check(cl_int err, char* clFunc, int print);
void read_program_source(char** programSource, const char* programName);
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queue, cl_context* contextPtr);
