
#include "header.h"

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
	vecadd_test(10000, &devices[0], &queueCPU, &context);
	vecadd_test(10000, &devices[1], &queueGPU, &context);
	
	
	
	// Clean-up
	clReleaseCommandQueue(queueCPU);
	clReleaseCommandQueue(queueGPU);
	clReleaseContext(context);
	
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
	
void error_check(cl_int err, char* clFunc, bool print)
{
	if(err != CL_SUCCESS)
	{
	    printf("Call to %s failed (%d) \n", clFunc, err);
	    exit(EXIT_FAILURE);
    }
	else if(print == 1)
	{
        printf("Call to %s success (%d) \n", clFunc, err);
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