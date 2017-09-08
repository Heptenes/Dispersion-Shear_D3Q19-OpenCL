int main(int argc, char *argv[])
{
	host_param_struct hostDat; // Data not accessed by kernels

	cl_device_id devices[2]; // One CPU, one GPU
	analyse_platform(devices, &hostDat);

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
	int returnLB = simulation_main(&hostDat, devices, &queueCPU, &queueGPU, &context);

	// Clean-up
	clReleaseCommandQueue(queueCPU);
	clReleaseCommandQueue(queueGPU);
	clReleaseContext(context);

	return 0;
}

int simulation_main(host_param_struct* hostDat, cl_device_id* devices, cl_command_queue* CPU_QueuePtr, cl_command_queue* GPU_QueuePtr,
	cl_context* contextPtr)
{
	// Initialise parameter structs
	int_param_struct intDat;
	flp_param_struct flpDat;
	kernel_struct kernelDat;
	zone_struct* zoneDat;

	printf("Int struct size: %lu\n", sizeof(intDat));
	printf("Flp struct size: %lu\n", sizeof(flpDat));

	// Assign data arrays, read input
	initialize_data(&intDat, &flpDat, hostDat);
	int paramErrors = parameter_checking(&intDat, &flpDat, hostDat);
	if (paramErrors > 0) {
		exit(EXIT_FAILURE);
	}

	// Read sphere surface discretization points
	cl_float4* spherePoints;
	sphere_discretization(&intDat, &flpDat, &spherePoints);

	// Build LB kernels
	create_LB_kernels(&intDat, &kernelDat, contextPtr, devices);

	// Some useful data sizes (cl functions often need size_t*)
	size_t numNodes = intDat.LatticeSize[0]*intDat.LatticeSize[1]*intDat.LatticeSize[2];
	size_t numParThreads = hostDat->DomainDecomp[0]*hostDat->DomainDecomp[1]*hostDat->DomainDecomp[2];
	size_t numSurfPoints = intDat.TotalSurfPoints;
	size_t pointWorkSize = intDat.PointsPerWorkGroup;

	size_t fDataSize = numNodes*LB_Q*sizeof(cl_float);
	size_t a3DataSize = numNodes*3*sizeof(cl_float);
	size_t parA3DataSize = intDat.NumParticles*3*sizeof(cl_float); // Array implementation
	size_t parV4DataSize = intDat.NumParticles*sizeof(cl_float4);  // Vector type implementation

	// --- HOST ARRAYS ---------------------------------------------------------
	// Lattice fields
	cl_float* f_h = (cl_float*)malloc(fDataSize);
	cl_float* u_h = (cl_float*)malloc(a3DataSize);
	cl_float* gpf_h = (cl_float*)malloc(a3DataSize*intDat.MaxSurfPointsPerNode);
	cl_int* countPoint_h = (cl_int*)malloc(numNodes*sizeof(cl_uint));
	cl_float* tau_p_h = (cl_float*)malloc(numNodes*sizeof(cl_float));
	// Particle arrays
	cl_float4* parKin = (cl_float4*)malloc(parV4DataSize*4); // x, vel, rot (quaternion), ang vel
	cl_float4* parForce = (cl_float4*)malloc(parV4DataSize*2); // Force and torque
	cl_float4* parFluidForce; // Needs to be allocated based on number of work groups per particle
	cl_float4* parFluidForceSum = (cl_float4*)calloc(numSurfPoints*2, sizeof(cl_float4)); // Check this
	//
	cl_int* threadMembers = (cl_int*)malloc(numParThreads*intDat.NumParticles*sizeof(cl_int)); // Max; could reduce mem
	cl_uint* numParInThread = (cl_uint*)calloc(numParThreads, sizeof(cl_uint));
	//
	cl_int* parsZone = (cl_int*)malloc(intDat.NumParticles*sizeof(cl_int));
	cl_int* zoneMembers; // To be malloc'ed in initialize_particle_zones
	cl_uint* numParInZone;

	// Initialization
	initialize_lattice_fields(hostDat, &intDat, &flpDat, f_h, gpf_h, u_h, tau_p_h, countPoint_h);
	initialize_particle_fields(hostDat, &intDat, &flpDat, parKin, parForce, &parFluidForce);
	initialize_particle_zones(hostDat, &intDat, &flpDat, parKin, parsZone, &zoneMembers, &numParInZone, threadMembers, numParInThread, &zoneDat);
	size_t totalNumZones = intDat.NumZones[0]*intDat.NumZones[1]*intDat.NumZones[2];

	// Stream mapping for pbcs
	cl_int* strMap;
	cl_int numPeriodicNodes = create_periodic_stream_mapping(&intDat, &strMap);
	printf("Periodic boundary nodes %d\n", numPeriodicNodes);
	size_t smDataSize = numPeriodicNodes*2*sizeof(cl_int);

	// --- CREATE BUFFERS --------------------------------------------------------
#define X(memName) cl_mem memName;
	LIST_OF_CL_MEM
#undef X
	// Lattice fields
	cl_int err_cl = CL_SUCCESS;
	fA_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	fB_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, fDataSize, NULL, NULL);
	u_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, a3DataSize, NULL, NULL);
	gpf_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, a3DataSize*intDat.MaxSurfPointsPerNode, NULL, NULL);
	countPoint_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, numNodes*sizeof(cl_int), NULL, NULL);
	if (intDat.ViscosityModel != 0) {
		tau_p_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, numNodes*sizeof(cl_float), NULL, NULL);
	}

	// Particle arrays
	parKin_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, parV4DataSize*4, parKin, NULL);
	parForce_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, parV4DataSize*2, parForce, NULL);
	parFluidForce_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, parV4DataSize*intDat.NumForceArrays*2, parFluidForce, NULL);
	parFluidForceSum_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE, numSurfPoints*sizeof(cl_float4)*2, NULL, NULL);
	parsZone_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, intDat.NumParticles*sizeof(cl_int), parsZone, NULL);
	zoneMembers_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, totalNumZones*intDat.NumParticles*sizeof(cl_int), zoneMembers, NULL);
	numParInZone_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, totalNumZones*sizeof(cl_int), numParInZone, NULL);

	// Read-only buffers
	intDat_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(int_param_struct), NULL, NULL);
	flpDat_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, sizeof(flp_param_struct), NULL, NULL);
	strMap_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, smDataSize, NULL, NULL);
	threadMembers_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, intDat.NumParticles*sizeof(cl_int), threadMembers, NULL);
	numParInThread_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, numParThreads*sizeof(cl_int), numParInThread, NULL);
	zoneDat_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, totalNumZones*sizeof(zone_struct), zoneDat, NULL);
	spherePoints_cl = clCreateBuffer(*contextPtr, CL_MEM_READ_ONLY, intDat.PointsPerParticle*sizeof(cl_float4), NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer 1", 1);

	// --- WRITE BUFFERS --------------------------------------------------------
	err_cl = CL_SUCCESS;
	err_cl = clEnqueueWriteBuffer(*GPU_QueuePtr, fA_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, fB_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, gpf_cl, CL_TRUE, 0, a3DataSize*intDat.MaxSurfPointsPerNode, gpf_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, countPoint_cl, CL_TRUE, 0, numNodes*sizeof(cl_int), countPoint_h, 0, NULL, NULL);
	if (intDat.ViscosityModel != 0) {
		err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, tau_p_cl, CL_TRUE, 0, numNodes*sizeof(cl_float), tau_p_h, 0, NULL, NULL);
	}
	error_check(err_cl, "clEnqueueWriteBuffer 1", 1);
	err_cl = CL_SUCCESS;
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, parFluidForceSum_cl, CL_TRUE, 0, numSurfPoints*sizeof(cl_float4)*2, parFluidForceSum, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, spherePoints_cl, CL_TRUE, 0, intDat.PointsPerParticle*sizeof(cl_float4), spherePoints, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, strMap_cl, CL_TRUE, 0, smDataSize, strMap, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, intDat_cl, CL_TRUE, 0, sizeof(intDat), &intDat, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(*GPU_QueuePtr, flpDat_cl, CL_TRUE, 0, sizeof(flpDat), &flpDat, 0, NULL, NULL);
	error_check(err_cl, "clEnqueueWriteBuffer 2", 1);

	// --- KERNEL RANGE SETTINGS -----------------------------------------------
	// Offset global id by 1, because of buffer layer
	size_t lattice_work_offset[3] = {1, 1, 1}; // Perf test this
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
	err_cl = CL_SUCCESS;

	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 2, memSize, &gpf_cl);
	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 3, memSize, &u_cl);
	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 4, memSize, &tau_p_cl);
	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 5, memSize, &countPoint_cl);
	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 6, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 7, memSize, &flpDat_cl);

	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 1, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 2, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 3, sizeof(cl_int), &wallAxis);

	err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 1, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 2, memSize, &strMap_cl);

	//cl_mem* pfflsMem[] = {&intDat_cl, &gpf_cl, &u_cl}; etc.
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 2, memSize, &gpf_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 3, memSize, &u_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 4, memSize, &parKin_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 5, memSize, &parFluidForce_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 6, memSize, &parFluidForceSum_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 7, memSize, &spherePoints_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_fluid_forces_linear_stencil, 8, memSize, &countPoint_cl);
			
	err_cl |= clSetKernelArg(kernelDat.sum_particle_fluid_forces, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.sum_particle_fluid_forces, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.sum_particle_fluid_forces, 2, memSize, &gpf_cl);
	
	err_cl |= clSetKernelArg(kernelDat.reset_particle_fluid_forces, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.reset_particle_fluid_forces, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.reset_particle_fluid_forces, 2, memSize, &gpf_cl);

	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 2, memSize, &parKin_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 3, memSize, &parForce_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 4, memSize, &zoneDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 5, memSize, &parsZone_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 6, memSize, &threadMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 7, memSize, &numParInThread_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 8, memSize, &zoneMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 9, memSize, &numParInZone_cl);

	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 2, memSize, &parKin_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 3, memSize, &parForce_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 4, memSize, &parFluidForce_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 5, memSize, &zoneDat_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 6, memSize, &parsZone_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 7, memSize, &threadMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 8, memSize, &numParInThread_cl);

	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 2, memSize, &parKin_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 3, memSize, &zoneDat_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 4, memSize, &threadMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 5, memSize, &numParInThread_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 6, memSize, &parsZone_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 7, memSize, &zoneMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 8, memSize, &numParInZone_cl);

	error_check(err_cl, "clSetKernelArg GPU kernels", 1);

	// ---------------------------------------------------------------------------------
	// --- MAIN LOOP -------------------------------------------------------------------
	// ---------------------------------------------------------------------------------
	FILE* vidPtr;
	vidPtr = fopen ("xyz_ovito_output.txt","w");
	printf("%s %d\n", "Starting iteration 1, maximum iterations", intDat.MaxIterations);
	for (int t=1; t<=intDat.MaxIterations; t++) {

		int toPrint = (t%hostDat->ConsolePrintFreq == 0) ? 1 : 0;

		if (toPrint) {
			printf("%s %d\n", "Starting iteration", t);
		}

		// Switch f buffers
		if (t%2 == 0) {
			err_cl  = clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 0, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 1, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 0, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 0, memSize, &fB_cl);
			//error_check(err_cl, "clSetKernelArg", 1);
		}
		else {
			err_cl  = clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 0, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.collideSRT_stream_D3Q19, 1, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 0, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 0, memSize, &fA_cl);
			//error_check(err_cl, "clSetKernelArg", 1);
		}

		// Kernel: LB collide and stream
		clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.collideSRT_stream_D3Q19, 3,
			lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);

		// Kernel: Particle update
		//clEnqueueNDRangeKernel(*CPU_QueuePtr, kernelDat.particle_dynamics, 1,
		//	NULL, &numParThreads, NULL, 0, NULL, NULL);

		clFinish(*GPU_QueuePtr);
		
		clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.reset_particle_fluid_forces, 3,
			lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);

		// Kernel: Periodic stream
		clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.boundary_periodic, 1,
			NULL, &periodic_work_size, NULL, 0, NULL, NULL);

		clFinish(*GPU_QueuePtr);

		// Kernel: LB velocity boundary
		if (velBoundary) {
			clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.boundary_velocity, 3,
				lattice_work_offset, velBC_work_size, NULL, 0, NULL, NULL);
			clFinish(*GPU_QueuePtr);
		}

		clFinish(*CPU_QueuePtr);
		// Need to write updated particle kinematics to GPU if not using host pointer

		// Kernel: Particle-fluid forces
		clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.particle_fluid_forces_linear_stencil, 1,
			NULL, &numSurfPoints, &pointWorkSize, 0, NULL, NULL);
		//printf("Checkpoint: particle_fluid_forces\n");
			
		clFinish(*CPU_QueuePtr);
		clEnqueueNDRangeKernel(*GPU_QueuePtr, kernelDat.sum_particle_fluid_forces, 3,
			lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);
			
		// Kernel: Particle-particle forces
		//clEnqueueNDRangeKernel(*CPU_QueuePtr, kernelDat.particle_particle_forces, 1,
		//	NULL, &numParThreads, NULL, 0, NULL, NULL);
		//printf("Checkpoint: particle_particle_forces\n");

		// Rebuild neighbour lists every intDat.RebuildFreq
		if (t%hostDat->RebuildFreq == 0) {
			for (int i = 0; i < totalNumZones; i++) {
				// Count is reset here because zones don't belong to threads
				// May impact performance if large num of zones
				numParInZone[i] = 0;
			}
			//clEnqueueNDRangeKernel(*CPU_QueuePtr, kernelDat.update_particle_zones, 1,
			//	NULL, &numParThreads, NULL, 0, NULL, NULL);
		}
		//printf("Checkpoint: update_particle_zones\n");

		clFinish(*GPU_QueuePtr); // So forces are updated
		clFinish(*CPU_QueuePtr);

		// Update dynamic parameters, e.g shear boundaries

		// Produce video output
		if (t%hostDat->VideoFreq == 0) {
			err_cl = clEnqueueReadBuffer(*GPU_QueuePtr, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
			error_check(err_cl, "clEnqueueReadBuffer Video", 1);
			
			continuous_output(hostDat, &intDat, u_h, parKin, vidPtr, t);
		}

	}
	printf("Checkpoint: end of simulation loop\n");

	// --- COPY DATA TO HOST ---------------------------------------------------
	// Velocity
	err_cl = clEnqueueReadBuffer(*GPU_QueuePtr, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
	error_check(err_cl, "clEnqueueReadBuffer", 1);

	write_lattice_field(u_h, &intDat);
	
	printf("Final force on particle 1 = %f %f %f\n", parFluidForce[0].x, parFluidForce[0].y, parFluidForce[0].z);

	// Cleanup
#define X(kernelName) clReleaseKernel(kernelDat.kernelName);
	LIST_OF_KERNELS
#undef X

#define X(memName) clReleaseMemObject(memName);
	LIST_OF_CL_MEM
#undef X

	return 0;
}
