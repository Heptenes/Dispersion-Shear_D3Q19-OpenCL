int main(int argc, char *argv[])
{
	int_param_struct intDat;
	flp_param_struct flpDat;
	kernel_struct kernelDat;
	host_param_struct hostDat; // Data not accessed by kernels
	output_data_struct outDat;
	outDat.ShearStressCount = 0;
	outDat.ShearStressAvg = 0.0f;
	outDat.ActualShearRate = 0.0f;
	
	printf("Int struct size: %lu\n", (unsigned long)sizeof(intDat));
	printf("Flp struct size: %lu\n", (unsigned long)sizeof(flpDat));

	cl_device_id deviceArr[2]; // One CPU, one GPU
	analyse_platform(deviceArr, &hostDat);

	// Create context and queues
	cl_int error;
	cl_context contextSim;
	contextSim = clCreateContext(NULL, 2, deviceArr, NULL, NULL, &error);
	error_check(error, "clCreateContext", 1);
	
	// Create programs
	cl_program programCPU, programGPU;
	//cl_program* programCPUptr = &programCPU;
	//cl_program* programGPUptr = &programGPU;

	// Create a command queue for CPU and GPU
	cl_command_queue queueCPU, queueGPU;

#ifdef __APPLE__
	queueCPU = clCreateCommandQueue(contextSim, deviceArr[0], 0, &error);
	error_check(error, "clCreateCommandQueue", 1);

	queueGPU = clCreateCommandQueue(contextSim, deviceArr[1], 0, &error);
	error_check(error, "clCreateCommandQueue", 1);
#else
	queueCPU = clCreateCommandQueueWithProperties(contextSim, deviceArr[0], 0, &error);
	error_check(error, "clCreateCommandQueue", 1);

	queueGPU = clCreateCommandQueueWithProperties(contextSim, deviceArr[1], 0, &error);
	error_check(error, "clCreateCommandQueue", 1);
#endif

	// Assign data arrays, read input
	initialize_data(&intDat, &flpDat, &hostDat);
	int paramErrors = parameter_checking(&intDat, &flpDat, &hostDat);
	if (paramErrors > 0) {
		exit(EXIT_FAILURE);
	}

	// Read sphere surface discretization points
	cl_float4* spherePoints;
	sphere_discretization(&intDat, &flpDat, &spherePoints);

	// Build LB kernels
	create_LB_kernels(&intDat, &kernelDat, &contextSim, deviceArr, &programCPU, &programGPU);

	// Some useful data sizes (cl functions often need size_t*)
	size_t numNodes = intDat.LatticeSize[0]*intDat.LatticeSize[1]*intDat.LatticeSize[2];
	//size_t splitKernelSize = 1 + (numNodes-1)/maxKernelSize;
	//printf("Splitting fluid kernel into %d kernels\n", (int)splitKernelSize);
	//size_t fluid_kernel_work_size[3];
	//size_t fluid_kernel_work_offset[3];

	size_t numParThreads = hostDat.DomainDecomp[0]*hostDat.DomainDecomp[1]*hostDat.DomainDecomp[2];
	size_t numSurfPoints = intDat.NumParticles > 0 ? intDat.TotalSurfPoints : 32;
	size_t pointWorkSize = intDat.PointsPerWorkGroup;

	size_t fDataSize = numNodes*19*sizeof(cl_float);
	size_t a3DataSize = numNodes*3*sizeof(cl_float);
	//size_t parA3DataSize = intDat.NumParticles*3*sizeof(cl_float); // Array implementation
	size_t parV4DataSize = intDat.NumParticles*sizeof(cl_float4);  // Vector type implementation

	// --- HOST ARRAYS ---------------------------------------------------------
	// Lattice fields
	cl_float* f_h = (cl_float*)malloc(fDataSize);
	cl_float* u_h = (cl_float*)malloc(a3DataSize);
	cl_float* gpf_h = (cl_float*)malloc(a3DataSize*intDat.MaxSurfPointsPerNode);
	cl_int* countPoint_h = (cl_int*)malloc(numNodes*sizeof(cl_int));
	cl_float* tau_lb_h = (cl_float*)malloc(numNodes*sizeof(cl_float));
	// Particle arrays
	cl_float4* parKin_h = (cl_float4*)malloc(parV4DataSize*4); // x, vel, rot (quaternion), ang vel
	cl_float4* parForce_h = (cl_float4*)malloc(parV4DataSize*2); // Force and torque
	cl_float4* parFluidForce_h = (cl_float4*)malloc(parV4DataSize*intDat.NumForceArrays*2);
	cl_float4* parFluidForceSum_h = (cl_float4*)calloc(numSurfPoints*2, sizeof(cl_float4)); // Check this
	//
	cl_int* threadMembers_h = (cl_int*)malloc(numParThreads*intDat.NumParticles*sizeof(cl_int)); // Max; could reduce mem
	cl_int* numParInThread_h = (cl_int*)calloc(numParThreads, sizeof(cl_int));
	//
	cl_int* parsZone_h = (cl_int*)malloc(intDat.NumParticles*sizeof(cl_int));
	cl_int* zoneMembers_h; // To be malloc'ed in initialize_particle_zones
	cl_int* numParInZone_h;
	cl_int* zoneNeighDat_h;

	// Initialization
	initialize_lattice_fields(&hostDat, &intDat, &flpDat, f_h, gpf_h, u_h, tau_lb_h, countPoint_h);
	initialize_particle_fields(&hostDat, &intDat, &flpDat, parKin_h, parForce_h, parFluidForce_h);
	initialize_particle_zones(&hostDat, &intDat, &flpDat, parKin_h, parsZone_h, &zoneMembers_h, &numParInZone_h, 
		threadMembers_h, numParInThread_h, &zoneNeighDat_h);
		
	size_t totalNumZones = intDat.NumZones[0]*intDat.NumZones[1]*intDat.NumZones[2];
	printf("Total num zones = %lu\n", (unsigned long)totalNumZones);
	
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
	fA_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, fDataSize, NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer fA", 1);
	fB_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, fDataSize, NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer fB", 1);
	u_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, a3DataSize, NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer u_cl", 1);
	gpf_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, a3DataSize*intDat.MaxSurfPointsPerNode, NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer gpf_cl", 1);
	countPoint_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, numNodes*sizeof(cl_int), NULL, &err_cl);
	tau_lb_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, numNodes*sizeof(cl_float), NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer 1", 1);

	// Particle arrays (host accessible)
	err_cl = CL_SUCCESS;
	parKin_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, parV4DataSize*4, parKin_h, &err_cl);
	parForce_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, parV4DataSize*2, parForce_h, &err_cl);
	parFluidForce_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, parV4DataSize*intDat.NumForceArrays*2, parFluidForce_h, &err_cl);
	parsZone_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, intDat.NumParticles*sizeof(cl_int), parsZone_h, &err_cl);
	zoneMembers_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, totalNumZones*intDat.NumParticles*sizeof(cl_int), zoneMembers_h, &err_cl);
	numParInZone_cl = clCreateBuffer(contextSim, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, totalNumZones*sizeof(cl_int), numParInZone_h, &err_cl);
	parFluidForceSum_cl = clCreateBuffer(contextSim, CL_MEM_READ_WRITE, numSurfPoints*sizeof(cl_float4)*2, NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer 2", 1);
	
	// Read-only buffers
	err_cl = CL_SUCCESS;
	threadMembers_cl = clCreateBuffer(contextSim,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, numParThreads*intDat.NumParticles*sizeof(cl_int), threadMembers_h, &err_cl);
	numParInThread_cl = clCreateBuffer(contextSim,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, numParThreads*sizeof(cl_int), numParInThread_h, &err_cl);
	zoneNeighDat_cl = clCreateBuffer(contextSim,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR, 28*totalNumZones*sizeof(cl_int), zoneNeighDat_h, &err_cl);
	error_check(err_cl, "clCreateBuffer 3", 1);
	
	intDat_cl = clCreateBuffer(contextSim, CL_MEM_READ_ONLY, sizeof(int_param_struct), NULL, &err_cl);
	flpDat_cl = clCreateBuffer(contextSim, CL_MEM_READ_ONLY, sizeof(flp_param_struct), NULL, &err_cl);
	strMap_cl = clCreateBuffer(contextSim, CL_MEM_READ_ONLY, smDataSize, NULL, &err_cl);
	spherePoints_cl = clCreateBuffer(contextSim, CL_MEM_READ_ONLY, intDat.PointsPerParticle*sizeof(cl_float4), NULL, &err_cl);
	error_check(err_cl, "clCreateBuffer 4", 1);

	// --- WRITE BUFFERS --------------------------------------------------------
	err_cl = CL_SUCCESS;
	err_cl = clEnqueueWriteBuffer(queueGPU, fA_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, fB_cl, CL_TRUE, 0, fDataSize, f_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, gpf_cl, CL_TRUE, 0, a3DataSize*intDat.MaxSurfPointsPerNode, gpf_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, countPoint_cl, CL_TRUE, 0, numNodes*sizeof(cl_int), countPoint_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, tau_lb_cl, CL_TRUE, 0, numNodes*sizeof(cl_float), tau_lb_h, 0, NULL, NULL);

	error_check(err_cl, "clEnqueueWriteBuffer 1", 1);
	err_cl = CL_SUCCESS;
	err_cl |= clEnqueueWriteBuffer(queueGPU, parFluidForceSum_cl, CL_TRUE, 0, numSurfPoints*sizeof(cl_float4)*2, parFluidForceSum_h, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, spherePoints_cl, CL_TRUE, 0, intDat.PointsPerParticle*sizeof(cl_float4), spherePoints, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, strMap_cl, CL_TRUE, 0, smDataSize, strMap, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, intDat_cl, CL_TRUE, 0, sizeof(intDat), &intDat, 0, NULL, NULL);
	err_cl |= clEnqueueWriteBuffer(queueGPU, flpDat_cl, CL_TRUE, 0, sizeof(flpDat), &flpDat, 0, NULL, NULL);
	error_check(err_cl, "clEnqueueWriteBuffer 2", 1);

	// --- KERNEL RANGE SETTINGS -----------------------------------------------
	int usingParticles = intDat.NumParticles > 0 ? 1 : 0;
	// Offset global id by 1, because of buffer layer
	size_t lattice_work_offset[3] = {1, 1, 1}; // Perf test this
	size_t global_work_size[3];
	size_t velBC_work_size[3];
	size_t tanBC_work_size[3];
	cl_int wallAxis=0; cl_int tanAxis=0;
	cl_int calcRho=0; cl_int tanCalcRho=0;

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
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 2, memSize, &gpf_cl);
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 3, memSize, &u_cl);
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 4, memSize, &tau_lb_cl);
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 5, memSize, &countPoint_cl);
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 6, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.collide_stream, 7, memSize, &flpDat_cl);

	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 1, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 2, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 3, sizeof(cl_int), &wallAxis);
	err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 4, sizeof(cl_int), &calcRho);

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
	err_cl |= clSetKernelArg(kernelDat.particle_particle_forces, 4, memSize, &zoneNeighDat_cl);
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
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 5, memSize, &threadMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.particle_dynamics, 6, memSize, &numParInThread_cl);

	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 0, memSize, &intDat_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 1, memSize, &flpDat_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 2, memSize, &parKin_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 3, memSize, &threadMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 4, memSize, &numParInThread_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 5, memSize, &parsZone_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 6, memSize, &zoneMembers_cl);
	err_cl |= clSetKernelArg(kernelDat.update_particle_zones, 7, memSize, &numParInZone_cl);

	error_check(err_cl, "clSetKernelArg GPU kernels", 1);

	// ---------------------------------------------------------------------------------
	// --- MAIN LOOP -------------------------------------------------------------------
	// ---------------------------------------------------------------------------------
	FILE* vidPtr;
	vidPtr = fopen ("xyz_ovito_output.txt","w");
	printf("%s %d\n", "Starting iteration 1, maximum iterations", intDat.MaxIterations);
	for (int t=1; t<=intDat.MaxIterations; t++) {

		int toPrint = (t%hostDat.ConsolePrintFreq == 0) ? 1 : 0;

		if (toPrint) {
			printf("%s %d\n", "Starting iteration", t);
		}

		// Switch f buffers
		if (t%2 == 0) {
			err_cl  = clSetKernelArg(kernelDat.collide_stream, 0, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.collide_stream, 1, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 0, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 0, memSize, &fB_cl);
			//error_check(err_cl, "clSetKernelArg", 1);
		}
		else {
			err_cl  = clSetKernelArg(kernelDat.collide_stream, 0, memSize, &fB_cl);
			err_cl |= clSetKernelArg(kernelDat.collide_stream, 1, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_velocity, 0, memSize, &fA_cl);
			err_cl |= clSetKernelArg(kernelDat.boundary_periodic, 0, memSize, &fA_cl);
			//error_check(err_cl, "clSetKernelArg", 1);
		}

		/* Kernel: LB collide and stream
		for (size_t i_k = 0; i_k < splitKernelSize; i_k++) {

			fluid_kernel_work_size[0] = global_work_size[0]/splitKernelSize;
			fluid_kernel_work_size[1] = global_work_size[1];
			fluid_kernel_work_size[2] = global_work_size[2];
			fluid_kernel_work_offset[0] = lattice_work_offset[0] + i_k*global_work_size[0]/splitKernelSize;
			fluid_kernel_work_offset[1] = lattice_work_offset[1];
			fluid_kernel_work_offset[2] = lattice_work_offset[2];
			printf("Fluid kernel with global size %d %d %d\n", (int)fluid_kernel_work_size[0], (int)fluid_kernel_work_size[1], (int)fluid_kernel_work_size[2]);
			printf("and work offset %d %d %d\n", (int)fluid_kernel_work_offset[0], (int)fluid_kernel_work_offset[1], (int)fluid_kernel_work_offset[2]);

			clEnqueueNDRangeKernel(queueGPU, kernelDat.collide_stream, 3,
				gluid_kernel_work_offset, fluid_kernel_work_size, NULL, 0, NULL, NULL);
		} */

		// Kernel: LB collide and stream
		clEnqueueNDRangeKernel(queueGPU, kernelDat.collide_stream, 3,
			lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);

		// Kernel: Particle update
		if (usingParticles) {
			clEnqueueNDRangeKernel(queueCPU, kernelDat.particle_dynamics, 1,
				NULL, &numParThreads, NULL, 0, NULL, NULL);

			clFinish(queueCPU);
			clFinish(queueGPU);

			// Kernel: Reset particle-fluid force array
			clEnqueueNDRangeKernel(queueGPU, kernelDat.reset_particle_fluid_forces, 3,
				lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);
		}

		// Kernel: Periodic stream
		clEnqueueNDRangeKernel(queueGPU, kernelDat.boundary_periodic, 1,
			NULL, &periodic_work_size, NULL, 0, NULL, NULL);

		clFinish(queueGPU);
		clFinish(queueCPU);

		// Kernel: LB velocity boundary
		if (velBoundary) {
			clEnqueueNDRangeKernel(queueGPU, kernelDat.boundary_velocity, 3,
				lattice_work_offset, velBC_work_size, NULL, 0, NULL, NULL);

			// Additional tangential velocity boundaries (experimental)
			for (int i = 0; i < 3; i++) {
				if (hostDat.TangentialVelBC[i] == 1) {
					//printf("Applying tangential velocity bounary on axis %d\n", i);
					tanBC_work_size[i] = 2;
					tanBC_work_size[(i+1)%3] = intDat.LatticeSize[(i+1)%3]-2;
					tanBC_work_size[(i+2)%3] = intDat.LatticeSize[(i+2)%3]-2;
					tanAxis = i;

					clSetKernelArg(kernelDat.boundary_velocity, 3, sizeof(cl_int), &tanAxis);
					clSetKernelArg(kernelDat.boundary_velocity, 4, sizeof(cl_int), &tanCalcRho);

					clEnqueueNDRangeKernel(queueGPU, kernelDat.boundary_velocity, 3,
						lattice_work_offset, tanBC_work_size, NULL, 0, NULL, NULL);
				}
			}
			clSetKernelArg(kernelDat.boundary_velocity, 3, sizeof(cl_int), &wallAxis);
			clSetKernelArg(kernelDat.boundary_velocity, 4, sizeof(cl_int), &calcRho);
			clFinish(queueGPU);
		}

		// Kernel: Particle-fluid forces
		if (usingParticles) {
			clEnqueueNDRangeKernel(queueGPU, kernelDat.particle_fluid_forces_linear_stencil, 1,
				NULL, &numSurfPoints, &pointWorkSize, 0, NULL, NULL);

			clFinish(queueGPU);

			// Kernel: Sum particle-fluid forces (acting on fluid)
			clEnqueueNDRangeKernel(queueGPU, kernelDat.sum_particle_fluid_forces, 3,
				lattice_work_offset, global_work_size, NULL, 0, NULL, NULL);

			// Kernel: Particle-particle forces
			clEnqueueNDRangeKernel(queueCPU, kernelDat.particle_particle_forces, 1,
				NULL, &numParThreads, NULL, 0, NULL, NULL);
		}

		// Rebuild neighbour lists every intDat.RebuildFreq
		if (usingParticles && t%hostDat.RebuildFreq == 0) {
			
			numParInZone_h = (cl_int*)clEnqueueMapBuffer(queueCPU, 
				numParInZone_cl, CL_TRUE, CL_MAP_WRITE, 0, totalNumZones*sizeof(cl_int), 0, NULL, NULL, &err_cl);
			error_check(err_cl, "clEnqueueMapBuffer", 1);
			
			for (int i = 0; i < totalNumZones; i++) {
				// Count is reset here because zones don't belong to threads
				// May impact performance if large num of zones
				numParInZone_h[i] = 0;
			}
			clEnqueueUnmapMemObject(queueCPU, numParInZone_cl, numParInZone_h, 0, NULL, NULL);
			clFinish(queueCPU);
			clEnqueueNDRangeKernel(queueCPU, kernelDat.update_particle_zones, 1,
				NULL, &numParThreads, NULL, 0, NULL, NULL);
			clFinish(queueCPU);
			
		}

		// Update dynamic parameters, e.g shear boundaries

		// Produce video output and/or analysis
		if (t%hostDat.VideoFreq == 0) {
			err_cl = clEnqueueReadBuffer(queueGPU, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
			err_cl = clEnqueueReadBuffer(queueCPU, parKin_cl, CL_TRUE, 0, parV4DataSize*4, parKin_h, 0, NULL, NULL);
			error_check(err_cl, "clEnqueueReadBuffer Video", 0);

			clFinish(queueGPU); 
			clFinish(queueCPU);
			continuous_output(&hostDat, &intDat, u_h, parKin_h, vidPtr, t);
		}
		if (t > 3*intDat.MaxIterations/4 && t%hostDat.ShearStressFreq == 0) {
			err_cl = clEnqueueReadBuffer(queueGPU, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
			err_cl = clEnqueueReadBuffer(queueGPU, tau_lb_cl, CL_TRUE, 0, numNodes*sizeof(cl_float), tau_lb_h, 0, NULL, NULL);
			error_check(err_cl, "clEnqueueReadBuffer Shear stress", 0);

			compute_shear_stress(&outDat, &hostDat, &intDat, &flpDat, u_h, tau_lb_h, t);

		}

	}
	clFinish(queueGPU); 
	clFinish(queueCPU);
	printf("Checkpoint: end of simulation loop\n");

	// --- COPY DATA TO HOST ---------------------------------------------------
	// Velocity
	err_cl = clEnqueueReadBuffer(queueGPU, u_cl, CL_TRUE, 0, a3DataSize, u_h, 0, NULL, NULL);
	error_check(err_cl, "clEnqueueReadBuffer", 1);

	write_lattice_field(u_h, &intDat);

	
	if (usingParticles) {
		
		parFluidForce_h = (cl_float4*)clEnqueueMapBuffer(queueCPU, 
			parFluidForce_cl, CL_TRUE, CL_MAP_READ, 0, parV4DataSize*intDat.NumForceArrays*2, 0, NULL, NULL, &err_cl);
		error_check(err_cl, "clEnqueueMapBuffer", 1);
		
		float finalForce[3] = {0.0, 0.0, 0.0};
		for (int i_fa = 0; i_fa < intDat.NumForceArrays; i_fa++) {
			finalForce[0] += parFluidForce_h[i_fa].x;
			finalForce[1] += parFluidForce_h[i_fa].y;
			finalForce[2] += parFluidForce_h[i_fa].z;
			printf("Final force += %f %f %f\n", parFluidForce_h[i_fa].x, parFluidForce_h[i_fa].y, parFluidForce_h[i_fa].z);
		}
		printf("Final force on particle 1 = %f %f %f\n", finalForce[0], finalForce[1], finalForce[2]);
		clEnqueueUnmapMemObject(queueCPU, parFluidForce_cl, parFluidForce_h, 0, NULL, NULL);
	} 
	clFinish(queueCPU);
	printf("Checkpoint: end of output\n"); 

	// Cleanup
#define X(kernelName) clReleaseKernel(kernelDat.kernelName);
	LIST_OF_KERNELS
#undef X

	printf("Checkpoint: released kernels\n");

#define X(memName) clReleaseMemObject(memName);
	LIST_OF_CL_MEM
#undef X 

	printf("Checkpoint: end of sim_main\n");
	
	
	/* Clean-up
	clReleaseCommandQueue(queueCPU);
	clReleaseCommandQueue(queueGPU);
	clReleaseProgram(programCPU);
	clReleaseProgram(programGPU);
	clReleaseDevice(deviceArr[0]);
	clReleaseDevice(deviceArr[1]);
	clReleaseContext(contextSim); */
	
	return 0;
}
