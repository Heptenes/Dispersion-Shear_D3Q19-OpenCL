
#include "D3Q19-OpenCL_header.h"

int main(int argc, char *argv[])
{  
	cl_device_id devices[2]; // One CPU, one GPU
	analyse_platform(devices);

	// Create context and queues
    cl_int error;
    cl_context context;
    context = clCreateContext(NULL, 2, devices, NULL, NULL, &error);
	error_check(error, "clCreateContext", 1);
	
    // Create a command queue for CPU and GPU
    cl_command_queue queueCPU, queueGPU;
	
    queueCPU = clCreateCommandQueue(context, devices[0], 0, &error);
    error_check(error, "clCreateCommandQueue", 1);
	
    queueGPU = clCreateCommandQueue(context, devices[1], 0, &error);
    error_check(error, "clCreateCommandQueue", 1);
	
	// Test devices
	vecadd_test(10, &devices[0], &queueCPU, &context);
	vecadd_test(10, &devices[1], &queueGPU, &context);
	
	// Run LB calculation 
	int returnLB = LB_main(devices, &queueCPU, &context);
	
	// Clean-up
	clReleaseCommandQueue(queueCPU);
	clReleaseCommandQueue(queueGPU);
	clReleaseContext(context);
	
	return 0;
}

int LB_main(cl_device_id* devices, cl_command_queue* queuePtr, cl_context* contextPtr)
{
	
	// Initialise parameter structs 
	int_param_struct intDat;
	float_param_struct flpDat;
	host_param_struct hostDat; // Params which remain in host memory
	
	printf("%s %lu %lu\n\n", "Size of structs on host: ", sizeof(intDat), sizeof(flpDat));
	
	// Assign data arrays, read input
	initialize_data(&intDat, &flpDat, &hostDat);
	
	int NumCells = intDat.LatticeSize[0]*intDat.LatticeSize[1]*intDat.LatticeSize[2];
	size_t fDataSize = NumCells*LB_Q*sizeof(cl_float);
	
	cl_float* f_h; // f in host memory
	f_h = (cl_float*)malloc(fDataSize);
	
	initialize_distribution(&hostDat, &intDat, f_h);
	
	printf("%s %d\n", "Total number of cells ", NumCells);
	
	create_stream_mapping(&intDat);
	
	// Build LB kernels
	cl_kernel kernelCPU[2], kernelGPU[2];
	create_LB_kernels(contextPtr, devices, kernelCPU, kernelGPU);
	
	// Setup and write buffers
	cl_mem fA_cl, fB_cl;
    fA_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	fB_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	//
	cl_mem intDat_cl, flpDat_cl;

	
	// --- MAIN LOOP ------------------------
	for (int t=0; t<intDat.MaxIterations; t++) {
		

		// Switch f arrays

	}
		
	
	return 0;
}

int create_stream_mapping(int_param_struct* intDat)
{
	// Propagation optimized data layout
	
	
	return 0;
}

int create_LB_kernels(cl_context* contextPtr, cl_device_id* devices, cl_kernel* kernelCPU, 
	cl_kernel* kernelGPU)
{
	char* programSourceCPU = NULL;
	char* programSourceGPU = NULL;
	const char programNameCPU[] = "CPU_program.cl";
	const char programNameGPU[] = "GPU_program.cl";
	
	cl_int error;
	cl_program programCPU, programGPU;
	
	// Read program source code
	read_program_source(&programSourceCPU, programNameCPU);
	read_program_source(&programSourceGPU, programNameGPU);
	
	// Create and build programs for devices
	programCPU = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSourceCPU, 
		NULL, &error);
	error_check(error, "clCreateProgramWithSource CPU", 1);
	
	clBuildProgram(programCPU, 1, &devices[0], NULL, NULL, &error);
	error_check(error, "cclBuildProgram CPU", 1);
	
	programGPU = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSourceGPU, 
		NULL, &error);
	error_check(error, "clCreateProgramWithSource GPU", 1);
	
	clBuildProgram(programGPU, 1, &devices[1], NULL, NULL, &error);
	error_check(error, "cclBuildProgram GPU", 1);
	
	// Select kernels from program
	kernelCPU[1] = clCreateKernel(programCPU, "CPU_sphere_collide", &error);
	if (!error_check(error, "clCreateKernel CPU", 1))		
		print_program_build_log(&programCPU, &devices[0]);
	
	kernelGPU[1] = clCreateKernel(programGPU, "GPU_newtonian_collide_stream_D3Q19_SRT", &error);
	if (!error_check(error, "clCreateKernel GPU 1", 1))		
		print_program_build_log(&programGPU, &devices[1]);
	
	kernelGPU[2] = clCreateKernel(programGPU, "GPU_boundary_velocity", &error);
	if (!error_check(error, "clCreateKernel GPU 2", 1))		
		print_program_build_log(&programGPU, &devices[1]);
	
	return 0;
}

int print_program_build_log(cl_program* program, cl_device_id* device)
{
	size_t logSize;
	char* buildLog;
	clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	
	buildLog = (char*)malloc(logSize);
	clGetProgramBuildInfo(*program, *device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
	
	printf("%s\n%s\n", "Program build log:", buildLog);
	exit(EXIT_FAILURE);
}

// Function to set up data arrays and read input file
int initialize_data(int_param_struct* intDat, float_param_struct* flpDat,
host_param_struct* hostDat)
{
	// Constant data
	const cl_int BasisVelD3Q19[19][3] = { { 0, 0, 0}, 
	{ 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
	{ 1, 1, 0}, { 1,-1, 0}, { 1, 0, 1}, { 1, 0,-1},
	{-1, 1, 0}, {-1,-1, 0}, {-1, 0, 1}, {-1, 0,-1},
	{ 0, 1, 1}, { 0, 1,-1}, { 0,-1, 1}, { 0,-1,-1} };
	
	const cl_float EqWeightsD3Q19[19] = {1./3., 
	1./18., 1./18., 1./18., 1./18., 1./18., 1./18., 
	1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.};
	
	memcpy(intDat->BasisVel, BasisVelD3Q19, sizeof(BasisVelD3Q19));
	memcpy(flpDat->EqWeights, EqWeightsD3Q19, sizeof(EqWeightsD3Q19));
	
	// Variable/input data
	FILE *ifp;
	ifp = fopen("input_file.txt", "r");
	
	input_data_struct inputDefaults[] = {
		{"iterations", TYPE_INT, &(intDat->MaxIterations), "1000"},
		{"constant_body_force", TYPE_FLOAT_3VEC, &(flpDat->ConstBodyForce), "0.0 0.0 0.0"},
		{"viscosity_model", TYPE_INT, &(intDat->ViscosityModel), "0"},
		{"lattice_size", TYPE_INT_3VEC, &(intDat->LatticeSize), "32 32 32"},
		{"initial_f", TYPE_STRING, &(hostDat->initialDist), "zero_vel"},
		{"boundary_conditions_xyz", TYPE_INT_3VEC, &(intDat->BoundaryConds), "0 0 0"},
		{"velocity_bc_upper", TYPE_FLOAT_3VEC, &(flpDat->VelUpper), "0.0 0.0 0.0"},
		{"velocity_bc_lower", TYPE_FLOAT_3VEC, &(flpDat->VelLower), "0.0 0.0 0.0"}
	};
	
	int inputDefaultSize = sizeof(inputDefaults)/sizeof(inputDefaults[0]);
		
	// Set defaults
	for (int p=0; p<inputDefaultSize; p++)
	{
		char defaultLine[WORD_STRING_SIZE] = {0};
		sprintf(defaultLine, "%s %s", inputDefaults[p].keyword, inputDefaults[p].defString);
		process_input_line(&defaultLine[0], inputDefaults, inputDefaultSize);
	}
		
	// Loop over lines
	int nLines=0, maxLines=256;
	char fLine[128];
	
    while(fgets(fLine, sizeof(fLine), ifp)!=NULL)
	{
		int fLineLength = strlen(fLine);
		nLines++;
		
		process_input_line(&fLine[0], inputDefaults, inputDefaultSize);
		fLine[0] = '\0';
    }
	
	display_input_params(intDat, flpDat);	
	
	fclose(ifp);
	
	return 0;
}

int initialize_distribution(host_param_struct* hostDat, int_param_struct* intDat,
	cl_float* f_h)
{
	printf("%s %s\n", "Initial distribution type ", hostDat->initialDist);
	
	int NumCells = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2];
	
	if (strcasestr(hostDat->initialDist, "zero_vel") != NULL)
	{   	
		float vel[3] = {0.0, 0.0, 0.0};
		float f_eq[19];
		
		// Write according to propagation optimized data layout
		for(int i_f=0; i_f<19; i_f++)
		{
			for(int i_c=0; i_c<NumCells; i_c++)
			{
				equilibrium_distribution_D3Q19(1.0, vel, f_eq);
				f_h[i_f*NumCells + i_c] = f_eq[i_f];
			}	
		}
	}
	else // Zero is default
	{
		float vel[3] = {0.0, 0.0, 0.0};
		float f_eq[19];
		
		// Write according to propagation optimized data layout
		for(int i_f=0; i_f<19; i_f++)
		{
			for(int i_c=0; i_c<NumCells; i_c++)
			{
				equilibrium_distribution_D3Q19(1.0, vel, f_eq);
				f_h[i_f*NumCells + i_c] = f_eq[i_f];
			}	
		}
	}
	return 0;
}

int equilibrium_distribution_D3Q19(float rho, float* vel, float* f_eq)
{
	float vx = vel[0];
	float vy = vel[1];
	float vz = vel[2];
	
	float vsq = vx*vx + vy*vy + vz*vz;
	
	// f_eq = w*rho*(1 + 3*v.c + 4.5*(v.c)^2 - 1.5*|v|^2)
	
	f_eq[0] = (rho/3.0)*(1.0 - 1.5*vsq);
	
	f_eq[1] = (rho/18.0)*(1.0 + 3.0*vx + 4.5*vx*vx - 1.5*vsq);
	f_eq[2] = (rho/18.0)*(1.0 - 3.0*vx + 4.5*vx*vx - 1.5*vsq);
	f_eq[3] = (rho/18.0)*(1.0 + 3.0*vy + 4.5*vy*vy - 1.5*vsq);
	f_eq[4] = (rho/18.0)*(1.0 - 3.0*vy + 4.5*vy*vy - 1.5*vsq);
	f_eq[5] = (rho/18.0)*(1.0 + 3.0*vz + 4.5*vz*vz - 1.5*vsq);
	f_eq[6] = (rho/18.0)*(1.0 - 3.0*vz + 4.5*vz*vz - 1.5*vsq);
	
	f_eq[7] = (rho/36.0)*(1.0 + 3.0*(vx+vy) + 4.5*(vx+vy)*(vx+vy) - 1.5*vsq);
	f_eq[8] = (rho/36.0)*(1.0 + 3.0*(vx-vy) + 4.5*(vx-vy)*(vx-vy) - 1.5*vsq);
	f_eq[9] = (rho/36.0)*(1.0 + 3.0*(vx+vz) + 4.5*(vx+vz)*(vx+vz) - 1.5*vsq);
	f_eq[10] = (rho/36.0)*(1.0 + 3.0*(vx-vz) + 4.5*(vx-vz)*(vx-vz) - 1.5*vsq);
	
	f_eq[11] = (rho/36.0)*(1.0 + 3.0*(-vx+vy) + 4.5*(-vx+vy)*(-vx+vy) - 1.5*vsq);
	f_eq[12] = (rho/36.0)*(1.0 + 3.0*(-vx-vy) + 4.5*(-vx-vy)*(-vx-vy) - 1.5*vsq);
	f_eq[13] = (rho/36.0)*(1.0 + 3.0*(-vx+vz) + 4.5*(-vx+vz)*(-vx+vz) - 1.5*vsq);
	f_eq[14] = (rho/36.0)*(1.0 + 3.0*(-vx-vz) + 4.5*(-vx-vz)*(-vx-vz) - 1.5*vsq);
	
	f_eq[15] = (rho/36.0)*(1.0 + 3.0*(vy+vz) + 4.5*(vy+vz)*(vy+vz) - 1.5*vsq);
	f_eq[16] = (rho/36.0)*(1.0 + 3.0*(vy-vz) + 4.5*(vy-vz)*(vy-vz) - 1.5*vsq);
	f_eq[17] = (rho/36.0)*(1.0 + 3.0*(-vy+vz) + 4.5*(-vy+vz)*(-vy+vz) - 1.5*vsq);
	f_eq[18] = (rho/36.0)*(1.0 + 3.0*(-vy-vz) + 4.5*(-vy-vz)*(-vy-vz) - 1.5*vsq);
	
	return 0;
}

int process_input_line(char* fLine, input_data_struct* inputDefaults, int inputDefaultSize)
{
	for (int l=0; l<inputDefaultSize; l++)
	{
		// Search for match
		char* searchPtr = strcasestr(fLine, (inputDefaults+l)->keyword);
		
		if (searchPtr)
		{
			printf("%s %s\n", "Found input file line ", (inputDefaults+l)->keyword);
			
			// Cast as appropriate data types 
			if ((inputDefaults+l)->dataType == TYPE_INT)
			{
				cl_int valueInt;
				sscanf(fLine, "%*s %d", &valueInt);
				memcpy( (inputDefaults+l)->varPtr, &valueInt, sizeof(cl_int));
			}
			else if ((inputDefaults+l)->dataType == TYPE_FLOAT)
			{
				cl_float valueFloat;
				sscanf(fLine, "%*s %f", &valueFloat);
				memcpy( (inputDefaults+l)->varPtr, &valueFloat, sizeof(cl_float));
			}
			else if ((inputDefaults+l)->dataType == TYPE_INT_3VEC)
			{
				cl_int value3VecInt[3];
				sscanf(fLine, "%*s %d %d %d", value3VecInt, value3VecInt+1, value3VecInt+2);
				memcpy( (inputDefaults+l)->varPtr, value3VecInt, sizeof(cl_float)*3);
			}
			else if ((inputDefaults+l)->dataType == TYPE_FLOAT_3VEC)
			{
				cl_float value3VecFloat[3];
				sscanf(fLine, "%*s %f %f %f", value3VecFloat, value3VecFloat+1, value3VecFloat+2);
				memcpy( (inputDefaults+l)->varPtr, value3VecFloat, sizeof(cl_float)*3);
			}
			else if ((inputDefaults+l)->dataType == TYPE_STRING)
			{
				char valueString[WORD_STRING_SIZE];
				sscanf(fLine, "%*s %s", valueString);
				memcpy( (inputDefaults+l)->varPtr, valueString, sizeof(valueString) );
			}
		}
		else
		{
		
		}
	}
	
	return 0;
}

int display_input_params(int_param_struct* intDat, float_param_struct* flpDat)
{
	printf("%s %d \n", "ViscosityModel = ", intDat->ViscosityModel);
	printf("%s %f %f %f \n", "ConstBodyForce = ", flpDat->ConstBodyForce[0], 
	flpDat->ConstBodyForce[1], flpDat->ConstBodyForce[2]);
	
	return 0;
}

// Function to lookup the OpenCL platform, and determine the CPU and GPU device to use
void analyse_platform(cl_device_id* devices)
{
    printf("Analysing platform... \n");
	
    cl_int err;

    // Platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);

    cl_platform_id *platforms = NULL;
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

    err = clGetPlatformIDs(numPlatforms, platforms, NULL);

    char *platformName = NULL;
    size_t size = 0;
    clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, platformName, &size);
    platformName = (char*)malloc(size);
    clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, platformName, NULL);

    printf("There is %i OpenCL platform(s) available, the default is %s \n\n", numPlatforms, platformName);
	
	// Process available devices
	cl_uint numCPUs;
	cl_uint numGPUs;
	
	// CPUs
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, NULL, &numCPUs);	
    cl_device_id *devicePtrCPU = NULL;
    devicePtrCPU = (cl_device_id*)malloc(numCPUs*sizeof(cl_device_id));   
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, numCPUs, devicePtrCPU, NULL);

	// GPUs
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numGPUs);
	
	if(numGPUs == 0)
	{
	    printf("Error: No GPU found \n\n");
	    exit(EXIT_FAILURE);
    }
	
    cl_device_id *devicePtrGPU = NULL;
    devicePtrGPU = (cl_device_id*)malloc(numCPUs*sizeof(cl_device_id));   
	err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numCPUs, devicePtrGPU, NULL);	

	// Print CPU information
	for(int i=0; i < numCPUs; i++)
	{
		char buf_name[1024];
		cl_uint buf_cu, buf_freq;
		cl_ulong buf_mem;

		printf("CPU info, num. %i\n", i);
		clGetDeviceInfo(devicePtrCPU[i], CL_DEVICE_NAME, sizeof(buf_name), buf_name, NULL);
		printf("DEVICE_NAME = %s\n", buf_name);
		clGetDeviceInfo(devicePtrCPU[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_cu), &buf_cu, NULL);
		printf("DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_cu);
		clGetDeviceInfo(devicePtrCPU[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_freq), &buf_freq, NULL);
		printf("DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_freq);
		clGetDeviceInfo(devicePtrCPU[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_mem), &buf_mem, NULL);
		printf("DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_mem);
		
		size_t buf_wi_size[3];
		clGetDeviceInfo(devicePtrCPU[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_wi_size), &buf_wi_size, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu %lu %lu \n\n", (unsigned long)buf_wi_size[0], (unsigned long)buf_wi_size[1], (unsigned long)buf_wi_size[2]);
	}	
	
	// Print GPU information
	for(int i=0; i < numGPUs; i++)
	{
		char buf_name[1024];
		cl_uint buf_cu, buf_freq;
		cl_ulong buf_mem;

		printf("GPU info, num. %i\n", i);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_NAME, sizeof(buf_name), buf_name, NULL);
		printf("DEVICE_NAME = %s\n", buf_name);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_cu), &buf_cu, NULL);
		printf("DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_cu);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_freq), &buf_freq, NULL);
		printf("DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_freq);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_mem), &buf_mem, NULL);
		printf("DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_mem);
		
		size_t buf_wi_size[3];
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_wi_size), &buf_wi_size, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu %lu %lu \n\n", (unsigned long)buf_wi_size[0], (unsigned long)buf_wi_size[1], (unsigned long)buf_wi_size[2]);
	}
	
	// Deal with double precision, and case where there is no GPU
	
	// Use default devices for now
	devices[0] = devicePtrCPU[0];
	devices[1] = devicePtrGPU[0];
	
	free(platforms);
	free(platformName);
	free(devicePtrCPU);
	free(devicePtrGPU);
}

void read_program_source(char** programSourcePtr, const char* programName)
{
	FILE* file = fopen(programName, "r");
	fseek(file, 0, SEEK_END);
	size_t programSize = ftell(file);
	rewind(file);

	*programSourcePtr = (char*)malloc(programSize + 1); // sizeof(char) == 1
	(*programSourcePtr)[programSize] = '\0';
	
	fread(*programSourcePtr, sizeof(char), programSize, file);
	
	FILE* outputFile = fopen("test_output.txt", "w");
	fprintf(outputFile, "%s", (*programSourcePtr));
	
	fclose(file);
}
	
int error_check(cl_int err, char* clFunc, bool print)
{
	if(err != CL_SUCCESS)
	{
		printf("Call to %s failed (%d):\n", clFunc, err);
		
		switch((int)err)
		{
			// Standard OpenCL error codes
			case -1: printf("%s\n", "CL_DEVICE_NOT_FOUND"); break;
			case -2: printf("%s\n", "CL_DEVICE_NOT_AVAILABLE"); break;
			case -3: printf("%s\n", "CL_COMPILER_NOT_AVAILABLE"); break;
			case -4: printf("%s\n", "CL_MEM_OBJECT_ALLOCATION_FAILURE"); break;
			case -5: printf("%s\n", "CL_OUT_OF_RESOURCES"); break;
			case -6: printf("%s\n", "CL_OUT_OF_HOST_MEMORY"); break;
			case -7: printf("%s\n", "CL_PROFILING_INFO_NOT_AVAILABLE"); break;
			case -8: printf("%s\n", "CL_MEM_COPY_OVERLAP"); break;
			case -9: printf("%s\n", "CL_IMAGE_FORMAT_MISMATCH"); break;
			case -10: printf("%s\n", "CL_IMAGE_FORMAT_NOT_SUPPORTED"); break;
			case -11: printf("%s\n", "CL_BUILD_PROGRAM_FAILURE"); break;
			case -12: printf("%s\n", "CL_MAP_FAILURE"); break;
			case -13: printf("%s\n", "CL_MISALIGNED_SUB_BUFFER_OFFSET"); break;
			case -14: printf("%s\n", "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"); break;
			case -15: printf("%s\n", "CL_COMPILE_PROGRAM_FAILURE"); break;
			case -16: printf("%s\n", "CL_LINKER_NOT_AVAILABLE"); break;
			case -17: printf("%s\n", "CL_LINK_PROGRAM_FAILURE"); break;
			case -18: printf("%s\n", "CL_DEVICE_PARTITION_FAILED"); break;
			case -19: printf("%s\n", "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"); break;		
			case -30: printf("%s\n", "CL_INVALID_VALUE"); break;
			case -31: printf("%s\n", "CL_INVALID_DEVICE_TYPE"); break;
			case -32: printf("%s\n", "CL_INVALID_PLATFORM"); break;
			case -33: printf("%s\n", "CL_INVALID_DEVICE"); break;
			case -34: printf("%s\n", "CL_INVALID_CONTEXT"); break;
			case -35: printf("%s\n", "CL_INVALID_QUEUE_PROPERTIES"); break;
			case -36: printf("%s\n", "CL_INVALID_COMMAND_QUEUE"); break;
			case -37: printf("%s\n", "CL_INVALID_HOST_PTR"); break;
			case -38: printf("%s\n", "CL_INVALID_MEM_OBJECT"); break;
			case -39: printf("%s\n", "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"); break;
			case -40: printf("%s\n", "CL_INVALID_IMAGE_SIZE"); break;
			case -41: printf("%s\n", "CL_INVALID_SAMPLER"); break;
			case -42: printf("%s\n", "CL_INVALID_BINARY"); break;
			case -43: printf("%s\n", "CL_INVALID_BUILD_OPTIONS"); break;
			case -44: printf("%s\n", "CL_INVALID_PROGRAM"); break;
			case -45: printf("%s\n", "CL_INVALID_PROGRAM_EXECUTABLE"); break;
			case -46: printf("%s\n", "CL_INVALID_KERNEL_NAME"); break;
			case -47: printf("%s\n", "CL_INVALID_KERNEL_DEFINITION"); break;
			case -48: printf("%s\n", "CL_INVALID_KERNEL"); break;
			case -49: printf("%s\n", "CL_INVALID_ARG_INDEX"); break;
			case -50: printf("%s\n", "CL_INVALID_ARG_VALUE"); break;
			case -51: printf("%s\n", "CL_INVALID_ARG_SIZE"); break;
			case -52: printf("%s\n", "CL_INVALID_KERNEL_ARGS"); break;
			case -53: printf("%s\n", "CL_INVALID_WORK_DIMENSION"); break;
			case -54: printf("%s\n", "CL_INVALID_WORK_GROUP_SIZE"); break;
			case -55: printf("%s\n", "CL_INVALID_WORK_ITEM_SIZE"); break;
			case -56: printf("%s\n", "CL_INVALID_GLOBAL_OFFSET"); break;
			case -57: printf("%s\n", "CL_INVALID_EVENT_WAIT_LIST"); break;
			case -58: printf("%s\n", "CL_INVALID_EVENT"); break;
			case -59: printf("%s\n", "CL_INVALID_OPERATION"); break;
			case -60: printf("%s\n", "CL_INVALID_GL_OBJECT"); break;
			case -61: printf("%s\n", "CL_INVALID_BUFFER_SIZE"); break;
			case -62: printf("%s\n", "CL_INVALID_MIP_LEVEL"); break;
			case -63: printf("%s\n", "CL_INVALID_GLOBAL_WORK_SIZE"); break;
			case -64: printf("%s\n", "CL_INVALID_PROPERTY"); break;
			case -65: printf("%s\n", "CL_INVALID_IMAGE_DESCRIPTOR"); break;
			case -66: printf("%s\n", "CL_INVALID_COMPILER_OPTIONS"); break;
			case -67: printf("%s\n", "CL_INVALID_LINKER_OPTIONS"); break;
			case -68: printf("%s\n", "CL_INVALID_DEVICE_PARTITION_COUNT"); break;
		}
		return 0;
	}
	else if (print == 1)
	{
        printf("Call to %s success (%d) \n", clFunc, err);
		return 1;
	}
	else
	{
		return 1;
	}
}

// Test function to check command queue and device are working
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queuePtr, cl_context* contextPtr)
{

    cl_int* A = NULL;
    cl_int* B = NULL;
    cl_int* C = NULL; 
    
    size_t dataSize = size*sizeof(cl_int);
    A = (cl_int*)malloc(dataSize);
    B = (cl_int*)malloc(dataSize);
    C = (cl_int*)malloc(dataSize);
	
    // Initialize the input data
    for(int i=0; i<size; i++)
	{
        A[i] = i;
        B[i] = i*i;
    }
	
	// Build test program and kernel
	char* programSource = NULL;
	const char programName[] = "test_program.cl";
	
	cl_int error;
	cl_program program;
	
	read_program_source(&programSource, programName);
	program = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSource, NULL, &error);
	clBuildProgram(program, 1, devicePtr, NULL, NULL, NULL);
	
	cl_kernel kernel;
	kernel = clCreateKernel(program, "vecadd", &error);
	error_check(error, "clCreateKernel", 1);
	
	// Setup and write buffers
	cl_mem A_d, B_d, C_d;
    A_d = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, dataSize, NULL, NULL);
    B_d = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, dataSize, NULL, NULL);
    C_d = clCreateBuffer(*contextPtr, CL_MEM_WRITE_ONLY, dataSize, NULL, NULL);
	
    error = clEnqueueWriteBuffer(*queuePtr, A_d, CL_TRUE, 0, dataSize, A, 0, NULL, NULL);
    error |= clEnqueueWriteBuffer(*queuePtr, B_d, CL_TRUE, 0, dataSize, B, 0, NULL, NULL);
    error_check(error, "clEnqueueWriteBuffer", 1);
	
    error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_d);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_d);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_d);
	error_check(error, "clSetKernelArg", 1);
		
	// Enqueue kernel
	size_t globalSize = size;
	size_t localSize = 1;
	
	error = clEnqueueNDRangeKernel(*queuePtr, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	error_check(error, "clEnqueueNDRangeKernel", 1);
	clFinish(*queuePtr);
	
	// Check results
	error = clEnqueueReadBuffer(*queuePtr, C_d, CL_TRUE, 0, dataSize, C, 0, NULL, NULL);
	error_check(error, "clEnqueueReadBuffer", 1);
	
	for(int i=0; i<size; i++)
	{
        printf("%i %i %i \n", A[i], B[i], C[i]);
    }
	
	// Clean-up
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(A_d);
	clReleaseMemObject(B_d);
	clReleaseMemObject(C_d);
	free(A);
	free(B);
	free(C);
	free(programSource);
}