#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "struct_header_host.h"

#define TYPE_INT 0
#define TYPE_FLOAT 1
#define TYPE_INT_3VEC 2
#define TYPE_FLOAT_3VEC 3
#define TYPE_FLOAT_4VEC 4
#define TYPE_STRING 5

#define BC_PERIODIC 0
#define BC_VELOCITY 1

#define VISC_NEWTONIAN 1
#define VISC_POWER_LAW 2
#define VISC_HB 3
#define VISC_CASSON 4

#define WORD_STRING_SIZE 128


#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

// X-macros
#define LIST_OF_KERNELS \
	X(collide_stream) \
	X(boundary_velocity) \
	X(boundary_periodic) \
	X(particle_fluid_forces_linear_stencil) \
	X(sum_particle_fluid_forces) \
	X(reset_particle_fluid_forces) \
	X(particle_dynamics) \
	X(particle_particle_forces) \
	X(update_particle_zones)


#define LIST_OF_CL_MEM \
	X(intDat_cl) \
	X(flpDat_cl) \
	X(fA_cl) \
	X(fB_cl) \
	X(u_cl) \
	X(gpf_cl) \
	X(countPoint_cl) \
	X(tau_lb_cl) \
	X(parKin_cl) \
	X(parForce_cl) \
	X(parFluidForce_cl) \
	X(parFluidForceSum_cl) \
	X(spherePoints_cl) \
	X(strMap_cl) \
	X(parsZone_cl) \
	X(zoneMembers_cl) \
	X(numParInZone_cl) \
	X(zoneNeighDat_cl) \
	X(threadMembers_cl) \
	X(numParInThread_cl) 


// Struct to contain information not accessed from within kernels
typedef struct {

	cl_int ConsolePrintFreq;
	char InitialDist[WORD_STRING_SIZE];
	cl_float InitialVel[3];
	cl_int InitialParticleDistribution;
	cl_float ParticleBuffer;
	cl_int InterpOrderIBM;
	cl_int DomainDecomp[3];
	cl_float ParticleDensity;
	size_t WorkItemSizes[3];
	size_t MaxWorkGroupSize;
	cl_int RebuildFreq;
	cl_int VideoFreq;
	cl_int FluidOutputSpacing;
	cl_int TangentialVelBC[3];
	cl_int ShearStressFreq;
	cl_float RandParticleShift;

} host_param_struct;


typedef struct {
#define X(kernelName) cl_kernel kernelName;
	LIST_OF_KERNELS
#undef X
} kernel_struct;


typedef struct {

	char keyword[WORD_STRING_SIZE];
	int dataType;
	void* varPtr;
	char defString[WORD_STRING_SIZE];

} input_data_struct;


typedef struct {

	int ShearStressCount;
	float ShearStressAvg;
	float ActualShearRate;

} output_data_struct;


//int simulation_main(host_param_struct* hostDat, cl_device_id* devices, cl_command_queue* CPU_QueuePtr, cl_command_queue* GPU_QueuePtr,
//	cl_context* contextPtr, cl_program* programCPU, cl_program* programGPU);
	
void particle_dynamics(int_param_struct* intDat, cl_float4* parKinematics_h, cl_float4* parForces_h);

int initialize_data(int_param_struct* intParams, flp_param_struct* floatParams, host_param_struct* hostParams);

int parameter_checking(int_param_struct* intDat, flp_param_struct* flpDat, host_param_struct* hostDat);

void initialize_lattice_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float* f_h, cl_float* gpf_h, cl_float* u_h, cl_float* tau_lb_h, cl_int* countPoint);

void initialize_particle_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float4* parKinematics, cl_float4* parForce, cl_float4* parFluidForce);

void initialize_particle_zones(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat, cl_float4* parKinematics, 
	cl_int* parsZone, cl_int** zoneMembers, cl_int** numParInZone, cl_int* threadMembers, cl_int* numParInThread, cl_int** zoneNeighDat);

int equilibrium_distribution_D3Q19(float rho, float* vel, float* f_eq);

float compute_tau(int viscosityModel, float srtII, cl_float NewtonianTau, cl_float* nonNewtonianParams);

int process_input_line(char* fLine, input_data_struct* inputDefaults, int inputDefaultSize);

void sphere_discretization(int_param_struct* intDat, flp_param_struct* flpDat, cl_float4** spherePoints);

int create_periodic_stream_mapping(int_param_struct* intDat, cl_int** strMapPtr);

int write_lattice_field(cl_float* u_h, int_param_struct* intDat);

void continuous_output(host_param_struct* hostDat, int_param_struct* intDat, cl_float* u_h, cl_float4* parKin, FILE* vidPtr, int frame);

void compute_shear_stress(output_data_struct* outDat, host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float* u_h, cl_float* tau_lb_h, int frame);

int create_LB_kernels(int_param_struct* intDat, kernel_struct* kernelDat, cl_context* contextPtr, cl_device_id* devices,
	cl_program* programCPU, cl_program* programGPU);

int display_input_params(int_param_struct* intParams, flp_param_struct* floatParams);

int print_program_build_log(cl_program* program, cl_device_id* device);

void analyse_platform(cl_device_id* devices, host_param_struct* hostDat);

int error_check(cl_int err, char* clFunc, int print);

void read_program_source(char** programSource, const char* programName);

void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queue, cl_context* contextPtr);
