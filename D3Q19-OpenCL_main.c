
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
	int returnLB = LB_main(devices, &queueCPU, &queueGPU, &context);
	
	// Clean-up
	clReleaseCommandQueue(queueCPU);
	clReleaseCommandQueue(queueGPU);
	clReleaseContext(context);
	
	return 0;
}

int LB_main(cl_device_id* devices, 
	cl_command_queue* CPU_QueuePtr, cl_command_queue* GPU_QueuePtr, 
	cl_context* contextPtr)
{
	// Initialise parameter structs 
	int_param_struct intDat;
	flp_param_struct flpDat;
	host_param_struct hostDat; // Params which remain in host memory
	kernel_struct kernelDat;
	
	// Check alignment
	printf("Int struct size: %lu\n", sizeof(intDat));
	printf("Flp struct size: %lu\n", sizeof(flpDat));
	
	// Assign data arrays, read input
	initialize_data(&intDat, &flpDat, &hostDat);
	
	int paramErrors = parameter_checking(&intDat, &flpDat);
	if (paramErrors > 0) {
		exit(EXIT_FAILURE);
	}
	
	int NumNodes = intDat.LatticeSize[0]*intDat.LatticeSize[1]*intDat.LatticeSize[2];
	
	// Create arrays on host
	size_t fDataSize = NumNodes*LB_Q*sizeof(cl_float);
	size_t v3DataSize = NumNodes*3*sizeof(cl_float);
	
	// --- HOST ARRAYS ---------------------------------------------------------
	cl_float *f_h, *u_h, *g_h; // Distribution function and velocity 
	f_h = (cl_float*)malloc(fDataSize);
	u_h = (cl_float*)malloc(v3DataSize);
	g_h = (cl_float*)malloc(v3DataSize);
	
	initialize_lattice_fields(&hostDat, &intDat, &flpDat, f_h, g_h, u_h);
	printf("Total number of nodes %d\n", NumNodes);
	
	// Stream mapping
	cl_int* strMap;
	cl_int numPeriodicNodes = create_periodic_stream_mapping(&intDat, &strMap); 
	printf("Periodic boundary nodes %d\n", numPeriodicNodes);
	size_t smDataSize = numPeriodicNodes*2*sizeof(cl_int);
	for (int i_map=0; i_map<numPeriodicNodes; i_map++) {
		//printf("i_1D, typeBC: %d, %d\n", strMap[i_map],strMap[i_map+numPeriodicNodes]);
	}
	
	// Build LB kernels
	create_LB_kernels(contextPtr, devices, &kernelDat);

	// --- CREATE ARRAYS & BUFFERS ---------------------------------------------
	cl_mem fA_cl, fB_cl, g_cl, u_cl;
	fA_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	fB_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	u_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, v3DataSize, NULL, NULL);
	g_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, v3DataSize, NULL, NULL);
	
	// Read-only buffers
	cl_mem intDat_cl, flpDat_cl, strMap_cl;
	intDat_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(int_param_struct), NULL, NULL);
	flpDat_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(flp_param_struct), NULL, NULL);
	strMap_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, smDataSize, NULL, NULL);
	
	// Write to device
	cl_int error_h = CL_SUCCESS; 
	error_h = clEnqueueWriteBuffer(*GPU_QueuePtr, fA_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, fB_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, g_cl, CL_TRUE, 0, v3DataSize, g_h, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, u_cl, CL_TRUE, 0, v3DataSize, u_h, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, strMap_cl, CL_TRUE, 0, smDataSize, strMap, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, intDat_cl, CL_TRUE, 0, sizeof(intDat), &intDat, 0, NULL, NULL);
	error_h |= clEnqueueWriteBuffer(*GPU_QueuePtr, flpDat_cl, CL_TRUE, 0, sizeof(flpDat), &flpDat, 0, NULL, NULL);
	error_check(error_h, "clEnqueueWriteBuffer", 1);
	
	// --- KERNEL RANGE SETTINGS -----------------------------------------------
	// Offset global id by 1, because of buffer layer
	size_t global_work_offset[3] = {1, 1, 1}; // Perf test this
	size_t global_work_size[3];
	size_t velBC_work_size[3];
	cl_int wallAxis=0;

	// Work sizes
	int velBoundary=0;
	for (int dim=0; dim<3; dim++) {
		global_work_size[dim] = intDat.LatticeSize[dim] - 2;
		
		if (intDat.BoundaryConds[dim] == 1) {
			velBC_work_size[dim] = 2; // This is the velocity boundary pair
			wallAxis = dim;
			velBoundary = 1;
		}
		else {
			velBC_work_size[dim] = intDat.LatticeSize[dim] - 2;
		}
	}
	if (velBoundary) {
		char xyz[4] = "XYZ\0";
		printf("%s %c\n", "Velocity BC applied to walls normal to axis", xyz[wallAxis]);
	}
	
	size_t periodic_work_size = numPeriodicNodes;
		
	// --- FIXED KERNEL ARGS ---------------------------------------------------
	size_t memSize = sizeof(cl_mem);
	error_h = CL_SUCCESS; 
	error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 2, memSize, &g_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 3, memSize, &u_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 4, memSize, &intDat_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 5, memSize, &flpDat_cl);
	
	error_h |= clSetKernelArg(kernelDat.GPU_boundary_velocity, 1, memSize, &intDat_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_boundary_velocity, 2, memSize, &flpDat_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_boundary_velocity, 3, sizeof(cl_int), &wallAxis);

	error_h |= clSetKernelArg(kernelDat.GPU_boundary_periodic, 1, memSize, &intDat_cl);
	error_h |= clSetKernelArg(kernelDat.GPU_boundary_periodic, 2, memSize, &strMap_cl);
	error_check(error_h, "clSetKernelArg", 1);
	
	// ---------------------------------------------------------------------------------
	// --- MAIN LOOP -------------------------------------------------------------------
	// ---------------------------------------------------------------------------------
	printf("%s %d\n", "Starting iteration 1, maximum iterations", intDat.MaxIterations);
	for (int t=1; t<=intDat.MaxIterations; t++) {
		
		int toPrint = (t%hostDat.consolePrintFreq == 0) ? 1 : 0;
		
		if (toPrint) {
			printf("%s %d\n", "Starting iteration", t);
		}
		
		// Switch f buffers
		if (t%2 == 0) {
		    error_h  = clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 0, memSize, &fA_cl);
		    error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 1, memSize, &fB_cl);
			error_h |= clSetKernelArg(kernelDat.GPU_boundary_velocity, 0, memSize, &fB_cl);
			error_h |= clSetKernelArg(kernelDat.GPU_boundary_periodic, 0, memSize, &fB_cl);
			//error_check(error_h, "clSetKernelArg", 1);
		}
		else {
		    error_h  = clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 0, memSize, &fB_cl);
		    error_h |= clSetKernelArg(kernelDat.GPU_collideSRT_stream_D3Q19, 1, memSize, &fA_cl);
			error_h |= clSetKernelArg(kernelDat.GPU_boundary_velocity, 0, memSize, &fA_cl);
			error_h |= clSetKernelArg(kernelDat.GPU_boundary_periodic, 0, memSize, &fA_cl);
			//error_check(error_h, "clSetKernelArg", 1);
		}

	    clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.GPU_collideSRT_stream_D3Q19, 3,
	    	global_work_offset, global_work_size, NULL, 0, NULL, NULL);
		
		clFinish(*GPU_QueuePtr);
		
	    clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.GPU_boundary_periodic, 1, 
			NULL, &periodic_work_size, NULL, 0, NULL, NULL);
			
		clFinish(*GPU_QueuePtr);
		
		if (velBoundary) {
		    clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.GPU_boundary_velocity, 3, 
				global_work_offset, velBC_work_size, NULL, 0, NULL, NULL);
				
			clFinish(*GPU_QueuePtr);
		}
	}
	
	// --- COPY DATA TO HOST ---------------------------------------------------
	// Velocity
	error_h = clEnqueueReadBuffer(*GPU_QueuePtr, u_cl, CL_TRUE, 0, v3DataSize, u_h, 0, NULL, NULL);
	error_check(error_h, "clEnqueueReadBuffer", 1);
	
	write_lattice_field(u_h, &intDat);
	
	// Cleanup 	
#define X(kernelName) clReleaseKernel(kernelDat.kernelName);
	LIST_OF_KERNELS
#undef X
	clReleaseMemObject(fA_cl);
	clReleaseMemObject(fB_cl);
	clReleaseMemObject(intDat_cl);
	clReleaseMemObject(flpDat_cl);
		
	return 0;
}

int create_periodic_stream_mapping(int_param_struct* intDat, cl_int** strMapPtr)
{
	
	int N_x = intDat->LatticeSize[0];
	int N_y = intDat->LatticeSize[1];
	int N_z = intDat->LatticeSize[2];
	
	int b = 1; // Buffer layer thickness
	
	// Constant coord for each face and its value, then upper range of loop for other 2 axis
	// x-axis on inner loop where possible
	const int faceSpec[6][6] = {
		{0,b,       2,N_z-b-2, 1,N_y-b-2}, // N-b-2 because edges/verts of face treated separately
		{0,N_x-b-1, 2,N_z-b-2, 1,N_y-b-2}, // Loops are inclusive (<=) upper range
		{1,b,       2,N_z-b-2, 0,N_x-b-2},
		{1,N_y-b-1, 2,N_z-b-2, 0,N_x-b-2},
		{2,b,       1,N_y-b-2, 0,N_x-b-2},
		{2,N_z-b-1, 1,N_y-b-2, 0,N_x-b-2} 
	};
	
	// Two constant coords for each edge and their value, then upper range of the edge axis
	const int edgeSpec[12][6] = {
		{0,b,       1,b,       2,N_z-b-2},
		{0,b,       1,N_y-b-1, 2,N_z-b-2},
		{0,b,       2,b,       1,N_y-b-2},
		{0,b,       2,N_z-b-1, 1,N_y-b-2},		
		{0,N_x-b-1, 1,b,       2,N_z-b-2},
		{0,N_x-b-1, 1,N_y-b-1, 2,N_z-b-2},
		{0,N_x-b-1, 2,b,       1,N_y-b-2},
		{0,N_x-b-1, 2,N_z-b-1, 1,N_y-b-2},
		{1,b,       2,b,       0,N_x-b-2},
		{1,b,       2,N_z-b-1, 0,N_x-b-2},
		{1,N_y-b-1, 2,b,       0,N_x-b-2},
		{1,N_y-b-1, 2,N_z-b-1, 0,N_x-b-2}
	};
	
	// Coords of vertexes
	const int vertSpec[8][3] = {
		{b,         b,         b      },
		{b,         b,         N_z-b-1},
		{b,         N_y-b-1,   b      },
		{b,         N_y-b-1,   N_z-b-1},
		{N_x-b-1,   b,         b      },
		{N_x-b-1,   b,         N_z-b-1},
		{N_x-b-1,   N_y-b-1,   b      },
		{N_x-b-1,   N_y-b-1,   N_z-b-1}
	};
	
    int* tempMapping;
    tempMapping = (int*)malloc(N_x*N_y*N_z*2*sizeof(int)); // Max possible size 
	int numPeriodicNodes = 0;
	
	// Write:
	// 1D index, face/edge/vertex type
	
	// Faces
	for (int face=0; face<6; face++){
		// If face of periodic boundary
		if (intDat->BoundaryConds[(int)(face/2)] == 0) { 
			
			for (int i=b+1; i<=faceSpec[face][3]; i++) {
				for (int j=b+1; j<=faceSpec[face][5]; j++) {
									
					// Coords of this node
					int r[3];
					r[faceSpec[face][0]] = faceSpec[face][1]; // The constant coord in plane
					r[faceSpec[face][2]] = i;
					r[faceSpec[face][4]] = j;
					// 1D index of node
					int i_1D = r[0] + N_x*(r[1] + r[2]*N_y);
					
					tempMapping[numPeriodicNodes*2    ] = i_1D; // 1D index of node in f array
					tempMapping[numPeriodicNodes*2 + 1] = face; // Boundary node type
						
					numPeriodicNodes++;
				}
			}	
		}
	}
	
	// Edges
	// Small perf saving possible, because edges of velocity boundaries don't need all unknowns 
	for (int edge=0; edge<12; edge++) {
		
		// If edge of two periodic boundaries
		if (intDat->BoundaryConds[edgeSpec[edge][0]] == 0 
		||  intDat->BoundaryConds[edgeSpec[edge][2]] == 0) {
			
			for (int i=b+1; i<=edgeSpec[edge][5]; i++) {
				
				// Coords of this node
				int r[3];
				r[edgeSpec[edge][0]] = edgeSpec[edge][1]; // Constant coords of edge
				r[edgeSpec[edge][2]] = edgeSpec[edge][3];
				r[edgeSpec[edge][4]] = i;
				
				// 1D index of node
				int i_1D = r[0] + N_x*(r[1] + r[2]*N_y);
				
				tempMapping[numPeriodicNodes*2    ] = i_1D; // 1D index of node f array
				tempMapping[numPeriodicNodes*2 + 1] = 6 + edge; // Boundary node type
				
				numPeriodicNodes++;
			}
		}
	}
	
	// Verticies (any boundaries periodic)
	if (intDat->BoundaryConds[0] == 0 
	||  intDat->BoundaryConds[1] == 0
	||  intDat->BoundaryConds[2] == 0) {
		
		for (int vert=0; vert<8; vert++){
			
			// Coords of this node
			int r[3];
			r[0] = vertSpec[vert][0]; // Constant coords of vert
			r[1] = vertSpec[vert][1];
			r[2] = vertSpec[vert][2];
			
			// 1D index of node
			int i_1D = r[0] + N_x*(r[1] + r[2]*N_y);
			
			tempMapping[numPeriodicNodes*2    ] = i_1D; // 1D index of node in f array
			tempMapping[numPeriodicNodes*2 + 1] = 18 + vert; // Boundary node type
			
			numPeriodicNodes++;
		}
	}
	
	// Copy to mapping array with thread-coalesced memory layout
	*strMapPtr = (cl_int*)malloc(numPeriodicNodes*2*sizeof(cl_int));
	for (int node=0; node<numPeriodicNodes; node++) {
		(*strMapPtr)[                   node] = tempMapping[node*2    ]; // 1D index
		(*strMapPtr)[numPeriodicNodes + node] = tempMapping[node*2 + 1]; // Boundary node type
	}
    
    free(tempMapping);
	
	return numPeriodicNodes;
}

int parameter_checking(int_param_struct* intDat, flp_param_struct* flpDat)
{
	if ((intDat->BoundaryConds[0]+intDat->BoundaryConds[1]+intDat->BoundaryConds[2]) > 1) {
		printf("Error: More than 1 pair of faces with velocity boundaries not yet supported.\n");
		return 1;
	}
	
	return 0;
}

int create_LB_kernels(cl_context* contextPtr, cl_device_id* devices, kernel_struct* kernelDat)
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
	
	// Build for both devices (for debugging)
	clBuildProgram(programCPU, 2, devices, NULL, NULL, &error);
	error_check(error, "cclBuildProgram CPU", 1);
	
	programGPU = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSourceGPU, 
		NULL, &error);
	error_check(error, "clCreateProgramWithSource GPU", 1);
	
	clBuildProgram(programGPU, 2, devices, NULL, NULL, &error);
	error_check(error, "cclBuildProgram GPU", 1);
	
	// Select kernels from program
	kernelDat->CPU_sphere_collide = clCreateKernel(programCPU, "CPU_sphere_collide", &error);
	if (!error_check(error, "clCreateKernel CPU", 1))		
		print_program_build_log(&programCPU, &devices[0]);
	
	kernelDat->GPU_collideSRT_stream_D3Q19 = 
		clCreateKernel(programGPU, "GPU_collideMRT_stream_D3Q19", &error);
	if (!error_check(error, "clCreateKernel GPU_collide_stream", 1))		
		print_program_build_log(&programGPU, &devices[1]);
	
	kernelDat->GPU_boundary_velocity = clCreateKernel(programGPU, "GPU_boundary_velocity", &error);
	if (!error_check(error, "clCreateKernel GPU_boundary_velocity", 1))		
		print_program_build_log(&programGPU, &devices[1]);
	
	kernelDat->GPU_boundary_periodic = clCreateKernel(programGPU, "GPU_boundary_periodic", &error);
	if (!error_check(error, "clCreateKernel GPU_boundary_periodic", 1))		
		print_program_build_log(&programGPU, &devices[1]);
	
	
	clReleaseProgram(programCPU);
	clReleaseProgram(programGPU);
	
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
int initialize_data(int_param_struct* intDat, flp_param_struct* flpDat,
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
		{"console_print_freq", TYPE_INT, &(hostDat->consolePrintFreq), "10"},
		{"constant_body_force", TYPE_FLOAT_3VEC, &(flpDat->ConstBodyForce), "0.0 0.0 0.0"},
		{"newtonian_tau", TYPE_FLOAT, &(flpDat->NewtonianTau), "0.8"},
		{"viscosity_model", TYPE_INT, &(intDat->ViscosityModel), "0"},
		{"total_lattice_size", TYPE_INT_3VEC, &(intDat->LatticeSize), "32 32 32"},
		{"initial_f", TYPE_STRING, &(hostDat->initialDist), "zero"},
		{"initial_vel", TYPE_FLOAT_3VEC, &(hostDat->initialVel), "0.0 0.0 0.0"},
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
	int nLines=0;
	char fLine[128];
	
    while(fgets(fLine, sizeof(fLine), ifp)!=NULL) {
		nLines++;	
		process_input_line(&fLine[0], inputDefaults, inputDefaultSize);
		fLine[0] = '\0';
    }
	
	display_input_params(intDat, flpDat);	
	
	fclose(ifp);
	
	return 0;
}

int initialize_lattice_fields(host_param_struct* hostDat, int_param_struct* intDat, 
	flp_param_struct* flpDat, cl_float* f_h, cl_float* g_h, cl_float* u_h)
{
	printf("%s %s\n", "Initial distribution type ", hostDat->initialDist);
	
	int NumNodes = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2];
	
	if (strstr(hostDat->initialDist, "poiseuille") != NULL)
	{   	
		// Do something
	}
	else if (strstr(hostDat->initialDist, "constant") != NULL)
	{   	
		float vel[3];
		float f_eq[19];
		vel[0] = hostDat->initialVel[0];
		vel[1] = hostDat->initialVel[1];
		vel[2] = hostDat->initialVel[2];
		
		equilibrium_distribution_D3Q19(1.0, vel, f_eq);
		
		// Use propagation-optimized data layouts
		for(int i_c=0; i_c<NumNodes; i_c++) {
			
			for(int i_f=0; i_f<19; i_f++) {
				f_h[i_c + i_f*NumNodes] = f_eq[i_f];
			}	
			u_h[i_c               ] = hostDat->initialVel[0];
			u_h[i_c +     NumNodes] = hostDat->initialVel[1];
			u_h[i_c +   2*NumNodes] = hostDat->initialVel[2];
			g_h[i_c               ] = flpDat->ConstBodyForce[0];
			g_h[i_c +     NumNodes] = flpDat->ConstBodyForce[1];
			g_h[i_c +   2*NumNodes] = flpDat->ConstBodyForce[2];
		}
	}
	else // Zero is default
	{
		float vel[3] = {0.0, 0.0, 0.0};
		float f_eq[19];
		equilibrium_distribution_D3Q19(1.0, vel, f_eq);
		
		// Use propagation-optimized data layouts
		for(int i_c=0; i_c<NumNodes; i_c++) {
			
			for(int i_f=0; i_f<19; i_f++) {
				f_h[i_c + i_f*NumNodes] = f_eq[i_f];
			}	
			u_h[i_c               ] = 0.0;
			u_h[i_c +     NumNodes] = 0.0;
			u_h[i_c +   2*NumNodes] = 0.0;
			g_h[i_c               ] = flpDat->ConstBodyForce[0];
			g_h[i_c +     NumNodes] = flpDat->ConstBodyForce[1];
			g_h[i_c +   2*NumNodes] = flpDat->ConstBodyForce[2];
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
		char* searchPtr = strstr(fLine, (inputDefaults+l)->keyword);
		
		if (searchPtr)
		{
			//printf("%s %s\n", "Found input file line ", (inputDefaults+l)->keyword);
			
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

int write_lattice_field(cl_float* field, int_param_struct* intDat)
{
	FILE* fPtr;
	fPtr = fopen ("velocity_field_final.txt","w");
		
	int n_C = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2];
	
	for(int i_x=1; i_x < intDat->LatticeSize[0]-1; i_x++) {
		for(int i_y=1; i_y < intDat->LatticeSize[1]-1; i_y++) {
			for(int i_z=1; i_z < intDat->LatticeSize[2]-1; i_z++) {
				
				int i_1D = i_x + intDat->LatticeSize[0]*(i_y + intDat->LatticeSize[1]*i_z);
				
				// Index, then velocity
				fprintf(fPtr, "%d %d %d ", i_x, i_y, i_z);
				fprintf(fPtr, "%9.7f %9.7f %9.7f\n", field[i_1D],  field[i_1D + n_C],  field[i_1D + 2*n_C]);
				
			}
		}
	}
	return 0;
}

int display_input_params(int_param_struct* intDat, flp_param_struct* flpDat)
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
	
	cl_int error = CL_SUCCESS;

	// Platforms
	cl_uint numPlatforms;
	error = clGetPlatformIDs(0, NULL, &numPlatforms);
	error_check(error, "clGetPlatformIDs", 1);

	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

	error = clGetPlatformIDs(numPlatforms, platforms, NULL);
	error_check(error, "clGetPlatformIDs", 1);

	char *platformName = NULL;
	size_t size = 0;
	error = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, platformName, &size);
	platformName = (char*)malloc(size);
	error |= clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, size, platformName, NULL);
	error_check(error, "clGetPlatformInfo", 1);
	if (error != CL_SUCCESS) { 
		exit(EXIT_FAILURE);
	}
	
	printf("There is %i OpenCL platform(s) available, the default is %s \n\n", numPlatforms, platformName);
	
	// Process available devices
	cl_uint numCPUs;
	cl_uint numGPUs;
	
	// CPUs
	error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0, NULL, &numCPUs);	
	cl_device_id *devicePtrCPU = NULL;
	devicePtrCPU = (cl_device_id*)malloc(numCPUs*sizeof(cl_device_id));   
	error |= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, numCPUs, devicePtrCPU, NULL);

	// GPUs
	error |= clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numGPUs);
	error_check(error, "clGetDeviceIDs", 1);
	if (error != CL_SUCCESS) { 
		exit(EXIT_FAILURE);
	} else if(numGPUs == 0) {
		printf("Error: No GPU found \n\n");
		exit(EXIT_FAILURE);
	}
	
	cl_device_id *devicePtrGPU = NULL;
	devicePtrGPU = (cl_device_id*)malloc(numGPUs*sizeof(cl_device_id));   
	error = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numGPUs, devicePtrGPU, NULL);	
	
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
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu %lu %lu \n\n", (unsigned long)buf_wi_size[0], 
			(unsigned long)buf_wi_size[1], (unsigned long)buf_wi_size[2]);
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
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(buf_mem), &buf_mem, NULL);
		printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = %llu\n", (unsigned long long)buf_mem);
		
		size_t buf_wi_size[3];
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(buf_wi_size), &buf_wi_size, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu %lu %lu \n\n", (unsigned long)buf_wi_size[0], 
			(unsigned long)buf_wi_size[1], (unsigned long)buf_wi_size[2]);
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
	FILE* file = fopen(programName, "rb");
	fseek(file, 0, SEEK_END);
	size_t programSize = ftell(file);
	rewind(file);

	*programSourcePtr = (char*)malloc(programSize + 1); // sizeof(char) == 1
	(*programSourcePtr)[programSize] = '\0';
	
	fread(*programSourcePtr, sizeof(char), programSize, file);
	
	//FILE* outputFile = fopen("test_output.txt", "w");
	//fprintf(outputFile, "%s", (*programSourcePtr));
	
	fclose(file);
}
	
int error_check(cl_int err, char* clFunc, int print)
{
	if(err != CL_SUCCESS) {
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
	else if (print == 1) {
        printf("Call to %s success (%d) \n", clFunc, err);
		return 1;
	}
	else {
		return 1;
	}
}

// Test function to check command queue and device are working
void vecadd_test(int size, cl_device_id* devicePtr, cl_command_queue* queuePtr, cl_context* contextPtr)
{
	char buf_name[1024];
	clGetDeviceInfo(*devicePtr, CL_DEVICE_NAME, sizeof(buf_name), buf_name, NULL);
	printf("vecadd_test, DEVICE_NAME = %s\n", buf_name);

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
	
    cl_int error = CL_SUCCESS;
	cl_program program;
	cl_kernel kernel;
    
	read_program_source(&programSource, programName);
	program = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSource, NULL, &error);
    error_check(error, "clCreateProgramWithSource", 1);
 
	error = clBuildProgram(program, 1, devicePtr, NULL, NULL, NULL);
	error_check(error, "clBuildProgram", 1);
    
	kernel = clCreateKernel(program, "vecadd", &error);
	if (!error_check(error, "clCreateKernel vecadd", 1))		
	print_program_build_log(&program, devicePtr);
	
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
	
	//for(int i=0; i<size; i++) {
	//    printf("%i %i %i \n", A[i], B[i], C[i]);
	//}
	
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