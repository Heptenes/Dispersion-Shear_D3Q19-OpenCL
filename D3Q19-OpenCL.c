#include "D3Q19-OpenCL_header.h"

#include "sim_main.c"

// Function to set up data arrays and read input file
int initialize_data(int_param_struct* intDat, flp_param_struct* flpDat, host_param_struct* hostDat)
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
		{"console_print_freq", TYPE_INT, &(hostDat->ConsolePrintFreq), "10"},
		{"constant_body_force", TYPE_FLOAT_3VEC, &(flpDat->ConstBodyForce), "0.0 0.0 0.0"},
		{"newtonian_tau", TYPE_FLOAT, &(flpDat->NewtonianTau), "1.0"},
		{"viscosity_model", TYPE_INT, &(intDat->ViscosityModel), "0"},
		{"viscosity_params", TYPE_FLOAT_4VEC, &(flpDat->ViscosityParams), "0.0 0.0 0.0 0.0"},
		{"total_lattice_size", TYPE_INT_3VEC, &(intDat->LatticeSize), "32 32 32"},
		{"domain_decomposition", TYPE_INT_3VEC, &(hostDat->DomainDecomp), "2 2 2"},
		{"lattice_buffer_size", TYPE_INT_3VEC, &(intDat->BufferSize), "1 1 1"},
		{"initial_f", TYPE_STRING, &(hostDat->InitialDist), "zero"},
		{"initial_vel", TYPE_FLOAT_3VEC, &(hostDat->InitialVel), "0.0 0.0 0.0"},
		{"boundary_conditions_xyz", TYPE_INT_3VEC, &(intDat->BoundaryConds), "0 0 0"},
		{"tangential_vel_bcs", TYPE_INT_3VEC, &(hostDat->TangentialVelBC), "0 0 0"},
		{"maintain_shear_rate", TYPE_INT_3VEC, &(intDat->MaintainShear), "0"},
		{"velocity_bc_upper", TYPE_FLOAT_3VEC, &(flpDat->VelUpper), "0.0 0.0 0.0"},
		{"velocity_bc_lower", TYPE_FLOAT_3VEC, &(flpDat->VelLower), "0.0 0.0 0.0"},
		{"num_particles", TYPE_INT, &(intDat->NumParticles), "0"},
		{"initial_particle_distribution", TYPE_INT, &(hostDat->InitialParticleDistribution), "1"},
		{"initial_particle_buffer", TYPE_FLOAT, &(hostDat->ParticleBuffer), "4.0"},
		{"particle_diameter", TYPE_FLOAT, &(flpDat->ParticleDiam), "8.0"},
		{"particle_density", TYPE_FLOAT, &(hostDat->ParticleDensity), "1.0"},
		{"particle_collision_model", TYPE_INT, &(intDat->ParForceModel), "1"},
		{"particle_collision_params", TYPE_FLOAT, &(flpDat->ParForceParams), "0.0, 0.0"},
		{"surf_point_write_atomic", TYPE_INT, &(intDat->MaxSurfPointsPerNode), "8"},
		{"ibm_interpolation_mode", TYPE_INT, &(hostDat->InterpOrderIBM), "1"},
		{"direct_forcing_coeff", TYPE_FLOAT, &(flpDat->DirectForcingCoeff), "1.0"},
		{"rebuild_neigh_list_freq", TYPE_INT, &(hostDat->RebuildFreq), "10"},
		{"video_freq", TYPE_INT, &(hostDat->VideoFreq), "1000"},
		{"shear_stress_freq", TYPE_INT, &(hostDat->ShearStressFreq), "1000"},
		{"fluid_ouput_spacing", TYPE_INT, &(hostDat->FluidOutputSpacing), "1"}
	};

	int inputDefaultSize = sizeof(inputDefaults)/sizeof(inputDefaults[0]);

	// Set defaults
	for (int p=0; p<inputDefaultSize; p++) {
		//
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

// Print out some important input information
int parameter_checking(int_param_struct* intDat, flp_param_struct* flpDat, host_param_struct* hostDat)
{
	int NumNodes = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2];
	printf("Total number of nodes: %d\n", NumNodes);

	printf("Particle domain decomposition: %dx%dx%d\n", hostDat->DomainDecomp[0], hostDat->DomainDecomp[1], hostDat->DomainDecomp[2]);

	printf("Viscosity model %d\n", intDat->ViscosityModel);

	if ((intDat->BoundaryConds[0]+intDat->BoundaryConds[1]+intDat->BoundaryConds[2]) > 1) {
		printf("Error: More than 1 pair of faces with velocity boundaries not yet supported.\n");
		return 1;
	}

	return 0;
}

void initialize_lattice_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
		cl_float* f_h, cl_float* gpf_h, cl_float* u_h, cl_float* tau_lb_h, cl_int* countPoint)
{
	printf("%s %s\n", "Initial distribution type ", hostDat->InitialDist);

	int NumNodes = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2];

	if (strstr(hostDat->InitialDist, "poiseuille") != NULL)
	{
		// Do something
		perror("poiseuille starting profile not supported yet");
	}
	else if (strstr(hostDat->InitialDist, "constant") != NULL)
	{
		float vel[3];
		float f_eq[19];
		vel[0] = hostDat->InitialVel[0];
		vel[1] = hostDat->InitialVel[1];
		vel[2] = hostDat->InitialVel[2];

		printf("Initializing f with constant velocity = %e %e %e\n", vel[0], vel[1], vel[2]);

		equilibrium_distribution_D3Q19(1.0, vel, f_eq);

		// Use propagation-optimized data layouts
		for(int i_n=0; i_n<NumNodes; i_n++) {

			for(int i_f=0; i_f<19; i_f++) {
				f_h[i_n + i_f*NumNodes] = f_eq[i_f];
			}
			u_h[i_n               ] = hostDat->InitialVel[0];
			u_h[i_n +     NumNodes] = hostDat->InitialVel[1];
			u_h[i_n +   2*NumNodes] = hostDat->InitialVel[2];
		}
	}
	else // Zero is default
	{
		float vel[3] = {0.0, 0.0, 0.0};
		float f_eq[19];
		equilibrium_distribution_D3Q19(1.0, vel, f_eq);

		// Use propagation-optimized data layouts
		for(int i_n=0; i_n<NumNodes; i_n++) {

			for(int i_f=0; i_f<19; i_f++) {
				f_h[i_n + i_f*NumNodes] = f_eq[i_f];
			}
			u_h[i_n               ] = 0.0f;
			u_h[i_n +     NumNodes] = 0.0f;
			u_h[i_n +   2*NumNodes] = 0.0f;
		}
	}

	// Other fields
	for(int i_n=0; i_n<NumNodes; i_n++) {

		tau_lb_h[i_n] = flpDat->NewtonianTau;
		countPoint[i_n] = 0;

		for (int p = 0; p < intDat->MaxSurfPointsPerNode; p++) {
			gpf_h[i_n + NumNodes*(3*p)    ] = 0.0f;
			gpf_h[i_n + NumNodes*(3*p + 1)] = 0.0f;
			gpf_h[i_n + NumNodes*(3*p + 2)] = 0.0f;
		}
	}

}

void initialize_particle_fields(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float4* parKinematics, cl_float4* parForce, cl_float4* parFluidForce)
{
	printf("\nThere are %d particles.\n",intDat->NumParticles);

	flpDat->ParticleMass = (4.0f/3.0f)*M_PI*hostDat->ParticleDensity*pow(flpDat->ParticleDiam,3);
	// 2*m*r^2/5;
	flpDat->ParticleMomInertia = 0.1f*flpDat->ParticleMass*flpDat->ParticleDiam*flpDat->ParticleDiam;
	printf("Mass = %f, moment of inertia = %f\n", flpDat->ParticleMass, flpDat->ParticleMomInertia);

	intDat->NumForceArrays = ceil((float)intDat->PointsPerParticle/(float)intDat->PointsPerWorkGroup);

	printf("\nNumber of particle force arrays needed = %d, (%d points, max %d per work group).\n\n",
		intDat->NumForceArrays, intDat->PointsPerParticle, intDat->PointsPerWorkGroup);

	int np = intDat->NumParticles;

	// Decide initial positions
	if (hostDat->InitialParticleDistribution == 1) {
		// NxNxN lattice, for approx. cubic systems
		float pb = hostDat->ParticleBuffer;
		int gridSize = ceil(pow(np,1.0/3.0));

		if (gridSize == 0) {
			perror("Error: InitialVel called without particles");
		}

		if (0.5*flpDat->ParticleDiam > pb) {
			perror("Error: Particles begin overlapping walls");
		}

		float sp[3]; // Spacing
		if (gridSize > 1) {
			for (int i = 0; i < 3; i++) {
				sp[i] = intDat->LatticeSize[i]-2*intDat->BufferSize[i]-1 - 2*pb;
				sp[i] /= (gridSize-1);
			}
		}
		else {
			for (int i = 0; i < 3; i++) {
				sp[i] = 0.0;
			}
		}

		for(int p = 0; p < np; p++) {
			cl_float px = pb + sp[0]*(p/(gridSize*gridSize));
			cl_float py = pb + sp[1]*((p/gridSize)%gridSize);
			cl_float pz = pb + sp[2]*(p%gridSize);
			// Position and velocity
			parKinematics[p     ] = (cl_float4){px, py, pz, 0.0f};
			parKinematics[p + np] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f};
		}
	}
	else if (hostDat->InitialParticleDistribution == 2){
		// NxNxM Lattice
		float pb = hostDat->ParticleBuffer;
		float wM = intDat->LatticeSize[2]-pb-1;
		float wN = intDat->LatticeSize[0]-pb-1;

		int mdn = ceil(wM/wN);

		int npl = 0, n = 0;
		while(npl < np) {
			n++;
			npl = n*n*n*mdn;
		}
		int m = n*mdn;
		printf("Choosing a %d x %d x %d lattice, max %d particles.\n", n, n, m, npl);

		float sp[3]; // Spacing
		if (np > 1) {
				sp[0] = (intDat->LatticeSize[0]-2*intDat->BufferSize[0]-1 - 2*pb)/(n-1);
				sp[1] = (intDat->LatticeSize[1]-2*intDat->BufferSize[1]-1 - 2*pb)/(n-1);
				sp[2] = (intDat->LatticeSize[2]-2*intDat->BufferSize[2]-1 - 2*pb)/(n*mdn-1);
		}
		else {
			for (int i = 0; i < 3; i++) {
				sp[i] = 0.0;
			}
		}

		printf("Lattice spacing = %f x %f x %f\n", sp[0], sp[1], sp[2]);

		for(int p = 0; p < np; p++) {
			cl_float px = pb + sp[0]*(p/(n*m));
			cl_float py = pb + sp[1]*((p/m)%n);
			cl_float pz = pb + sp[2]*(p%m);
			// Position and velocity
			parKinematics[p     ] = (cl_float4){px, py, pz, 0.0f};
			parKinematics[p + np] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f};
		}
	}
	else if (hostDat->InitialParticleDistribution == 3) {
		// Place particle centrally, wrt y and z
		// hostDat->ParticleBuffer along the x direction

		if (intDat->NumParticles > 1) {
			perror("Error: Single-particle initial distribution chosen with more than 1 particle.");
		}
		// Position and velocity
		float px = hostDat->ParticleBuffer;
		float py = ((float)intDat->LatticeSize[1]-3)/2.0f;
		float pz = ((float)intDat->LatticeSize[2]-3)/2.0f;
		parKinematics[0] = (cl_float4){{px, py, pz, 0.0f}};
		parKinematics[1] = (cl_float4){{0.0f, 0.0f, 0.0f, 0.0f}};
		printf("Placing single particle at position %f %f %f\n", px, py, pz);

	}
	else {
		perror("Error: initial_particle_distribution option not a known value\n");
	}

	// Angular quantities, forces and torque
	for(int p = 0; p < np; p++) {
		// Particle quaternion, last element is scalar part - easiest for use with float4 vectors
		parKinematics[p + 2*np] = (cl_float4){0.0f, 0.0f, 0.0f, 1.0f};
		// Angular velocity
		parKinematics[p + 3*np] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f};

		parForce[p     ] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f};
		parForce[p + np] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f};

		for (int fa = 0; fa < intDat->NumForceArrays; fa++) {
			parFluidForce[p + np*(2*fa)    ] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f}; // Force
			parFluidForce[p + np*(2*fa + 1)] = (cl_float4){0.0f, 0.0f, 0.0f, 0.0f}; // Torque
		}
	}

}

void initialize_particle_zones(host_param_struct* hostDat, int_param_struct* intDat, flp_param_struct* flpDat,
	cl_float4* parKinematics, cl_int* parsZone, cl_int** zoneMembers, cl_uint** numParInZone, cl_int* threadMembers, cl_uint* numParInThread,
	zone_struct** zoneDat)
{
	// Calculate estimate of max particle relative speed
	float vMax = 0.05f; // Guess
	for (int i = 0; i < 3; i++) {
		vMax = vMax < fabsf(flpDat->VelUpper[i]) ? fabsf(flpDat->VelUpper[i]) : vMax;
		vMax = vMax < fabsf(flpDat->VelLower[i]) ? fabsf(flpDat->VelLower[i]) : vMax;
		float nu = (flpDat->NewtonianTau-0.5f)/3.0f;
		float vMaxNewt = flpDat->ConstBodyForce[i]*(intDat->LatticeSize[i]-3.0f)*(intDat->LatticeSize[i]-3.0f)/(12.0f*nu);
		vMax = vMax < vMaxNewt ? vMaxNewt : vMax;
	}
	printf("\nEstimated max particle velocity = %f\n", vMax);
	if (intDat->ViscosityModel != 0) {
		printf("Warning: Estimate maybe inaccurate for forced flow of non-Newtonian fluids\n");
	}

	// Compute neighbor zone width
	float dMax = flpDat->ParticleDiam; // Monodisperse for now
	float minZoneWidth = 2*(vMax*hostDat->RebuildFreq) + dMax;
	printf("\nMinimum particle neighbor zone width = %f\n", minZoneWidth);

	// Compute number of zones in each dimension
	for (int i = 0; i < 3; i++) {
		// Total size in i'th dimension
		float w = (float)(intDat->LatticeSize[i]-2*intDat->BufferSize[i]-1);
		intDat->NumZones[i] = floor(w/minZoneWidth) > 0 ? floor(w/minZoneWidth) : 1;
		flpDat->ZoneWidth[i] = w/intDat->NumZones[i];
		printf("Number of particle zones in dimension %d = %d\n", i, intDat->NumZones[i]);
	}
	printf("\nActual particle neighbor zone width = %f x %f x %f\n", flpDat->ZoneWidth[0], flpDat->ZoneWidth[1], flpDat->ZoneWidth[2]);

	// Assign particles to threads and zones
	int numParThreads = hostDat->DomainDecomp[0]*hostDat->DomainDecomp[1]*hostDat->DomainDecomp[2];
	int totalNumZones = intDat->NumZones[0]*intDat->NumZones[1]*intDat->NumZones[2];
	*zoneDat = (zone_struct*)malloc(totalNumZones*sizeof(zone_struct));

	printf("Total number of zones = %d\n", totalNumZones);

	*zoneMembers = (cl_int*)malloc(totalNumZones*intDat->NumParticles*sizeof(cl_int));
	*numParInZone = (cl_uint*)calloc(totalNumZones, sizeof(cl_uint));

	for (int p = 0; p < intDat->NumParticles; p++) {

		// Particles always belong to their initial thread
		int thread = (int)(numParThreads*p)/intDat->NumParticles;
		int np = numParInThread[thread]++;
		threadMembers[thread*intDat->NumParticles + np] = p;
		printf("Particle %d belongs to thread %d\n", p, thread);
		printf("Thread %d now has %d particles.\n", thread, np+1);

		printf("Particle %d has position ", p);
		printf("%f %f %f\n", parKinematics[p].x, parKinematics[p].y, parKinematics[p].z);

		int zoneIDx = (int)(parKinematics[p].x/flpDat->ZoneWidth[0]);
		int zoneIDy = (int)(parKinematics[p].y/flpDat->ZoneWidth[1]);
		int zoneIDz = (int)(parKinematics[p].z/flpDat->ZoneWidth[2]);
		int zoneID = zoneIDx + intDat->NumZones[0]*(zoneIDy + intDat->NumZones[2]*zoneIDz);

		parsZone[p] = zoneID;
		(*zoneMembers)[zoneID*intDat->NumParticles + (*numParInZone)[zoneID]++] = p;
		//
		printf("Particle %d is %d'th particle of zone %d (%d,%d,%d).\n", p, (*numParInZone)[zoneID],
			zoneID, zoneIDx, zoneIDy, zoneIDz);

	}

	// Loop over zones and add neighbors (x on inner loop to match GPU arrays)
	for (int k = 0; k < intDat->NumZones[2]; k++) {
		//
		for (int j = 0; j < intDat->NumZones[1]; j++) {
			//
			for (int i = 0; i < intDat->NumZones[0]; i++) {

				int zoneID = i + intDat->NumZones[0]*(j + intDat->NumZones[2]*k);
				(*zoneDat)[zoneID].NumNeighbors = 0;

				int lm = -1; int lp = 1; int mm = -1; int mp = 1; int nm = -1; int np = 1;

				// Consider type 1 velocity BCs to be non-periodic, so don't add zones in that direction
				if (intDat->BoundaryConds[0] == 1 && i == 0) {lm = 0;}
				if (intDat->BoundaryConds[1] == 1 && j == 0) {mm = 0;}
				if (intDat->BoundaryConds[2] == 1 && k == 0) {nm = 0;}
				if (intDat->BoundaryConds[0] == 1 && i == intDat->NumZones[0]-1) {lp = 0;}
				if (intDat->BoundaryConds[1] == 1 && j == intDat->NumZones[1]-1) {mp = 0;}
				if (intDat->BoundaryConds[2] == 1 && k == intDat->NumZones[2]-1) {np = 0;}

				for (int n = nm; n <= np; n++) {
					//
					for (int m = mm; m <= mp; m++) {
						//
						for (int l = lm; l <= lp; l++) {

							// Shifted to neighbor location
							int ls = i+l; int ms = j+m; int ns = k+n;

							// Wrapped around boundaries
							int lw = ls < 0 ? intDat->NumZones[0]-1 : ls%intDat->NumZones[0];
							int mw = ms < 0 ? intDat->NumZones[1]-1 : ms%intDat->NumZones[1];
							int nw = ns < 0 ? intDat->NumZones[2]-1 : ns%intDat->NumZones[2];

							int neighID = lw + intDat->NumZones[0]*(mw + intDat->NumZones[2]*(nw));

							//printf("Zone %d (%d,%d,%d) has neighbor %d (%d,%d,%d)\n", zoneID,i,j,k, neighID,lw,mw,nw);
							int numNeighs = (*zoneDat)[zoneID].NumNeighbors++;
							(*zoneDat)[zoneID].NeighborZones[numNeighs] = neighID;
						}
					}
				}
			}
		}
	}
}

int create_LB_kernels(int_param_struct* intDat, kernel_struct* kernelDat, cl_context* contextPtr, cl_device_id* devices)
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
	programGPU = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSourceGPU,
		NULL, &error);
	error_check(error, "clCreateProgramWithSource GPU", 1);

	clBuildProgram(programGPU, 1, &devices[1], NULL, NULL, &error);
	error_check(error, "clBuildProgram GPU", 1);

	// CPU
	programCPU = clCreateProgramWithSource(*contextPtr, 1, (const char**)&programSourceCPU,
		NULL, &error);
	error_check(error, "clCreateProgramWithSource CPU", 1);

	clBuildProgram(programCPU, 1, &devices[0], NULL, NULL, &error);
	error_check(error, "clBuildProgram CPU", 1);

	// Select kernels from program
	// GPU
	kernelDat->collide_stream = clCreateKernel(programGPU, "collideMRT_stream_D3Q19", &error);
	if (!error_check(error, "clCreateKernel collideMRT_stream_D3Q19", 1))
		print_program_build_log(&programGPU, &devices[1]);

	kernelDat->boundary_velocity = clCreateKernel(programGPU, "boundary_velocity", &error);
	if (!error_check(error, "clCreateKernel boundary_velocity", 1))
		print_program_build_log(&programGPU, &devices[1]);

	kernelDat->boundary_periodic = clCreateKernel(programGPU, "boundary_periodic", &error);
	if (!error_check(error, "clCreateKernel boundary_periodic", 1))
		print_program_build_log(&programGPU, &devices[1]);

	kernelDat->particle_fluid_forces_linear_stencil = clCreateKernel(programGPU, "particle_fluid_forces_linear_stencil", &error);
	if (!error_check(error, "clCreateKernel fluid_particle_forces_linear_stencil", 1))
		print_program_build_log(&programGPU, &devices[1]);

	kernelDat->sum_particle_fluid_forces = clCreateKernel(programGPU, "sum_particle_fluid_forces", &error);
	if (!error_check(error, "clCreateKernel sum_particle_fluid_forces", 1))
		print_program_build_log(&programGPU, &devices[1]);

	kernelDat->reset_particle_fluid_forces = clCreateKernel(programGPU, "reset_particle_fluid_forces", &error);
	if (!error_check(error, "clCreateKernel reset_particle_fluid_forces", 1))
		print_program_build_log(&programGPU, &devices[1]);

	// CPU
	kernelDat->particle_dynamics = clCreateKernel(programCPU, "particle_dynamics", &error);
	if (!error_check(error, "clCreateKernel particle_dynamics", 1))
		print_program_build_log(&programCPU, &devices[0]);

	kernelDat->particle_particle_forces = clCreateKernel(programCPU, "particle_particle_forces", &error);
	if (!error_check(error, "clCreateKernel particle_particle_forces", 1))
		print_program_build_log(&programCPU, &devices[0]);

	kernelDat->update_particle_zones = clCreateKernel(programCPU, "update_particle_zones", &error);
	if (!error_check(error, "clCreateKernel update_particle_zones", 1))
		print_program_build_log(&programCPU, &devices[0]);

	size_t actualWorkGrpSize;
	clGetKernelWorkGroupInfo(kernelDat->particle_fluid_forces_linear_stencil, devices[1],
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &actualWorkGrpSize, NULL);

	int tempSize = (cl_int)actualWorkGrpSize;
	int power2 = (int)ceil(log2(tempSize)-1E-6);
	int workSize = 1;	// Integer exponentiation 2^power2
	for(int i = 0; i < power2; i++) {
		workSize *= 2;
	}
	printf("Max work group size of fluid-particle kernel = %d.\n", workSize);

	intDat->PointsPerWorkGroup = workSize > intDat->PointsPerParticle ? intDat->PointsPerParticle : workSize;



	clReleaseProgram(programCPU);
	clReleaseProgram(programGPU);

	return 0;
}

void sphere_discretization(int_param_struct* intDat, flp_param_struct* flpDat, cl_float4** spherePoints)
{
	// Calculated surface area
	float d = flpDat->ParticleDiam;
	float area = M_PI*d*d;
	int numPoints = 1;
	int power2 = ceil(log2(area));

	// Integer exponentiation 2^power2
	for(int i = 0; i < power2; i++) {
		numPoints *= 2;
	}
	numPoints = numPoints < 128 ? 128 : numPoints;
	numPoints = numPoints > 1024 ? 1024 : numPoints;

	printf("Number of surface points per particle = %d.\n", numPoints);
	*spherePoints = (cl_float4*)malloc(numPoints*sizeof(cl_float4));
	intDat->PointsPerParticle = numPoints;
	intDat->TotalSurfPoints = numPoints*intDat->NumParticles;
	flpDat->PointArea = area/(float)numPoints;
	printf("Surface area per point = %f.\n", flpDat->PointArea);

	// Take discretization with num nodes > surface area in lattice units
	char sphereFilename[128];
	char sphereFolder[] = "unit_sphere_partitions/";
	sprintf(sphereFilename, "%s%s%d%s", sphereFolder, "unit_sphere_partition_", numPoints, ".txt");
	printf("Reading sphere discretization from %s\n", sphereFilename);

	FILE* fs;
	fs = fopen(sphereFilename, "r");
	if (fs == NULL) {
		printf("%s not opened\n", sphereFilename);
	    perror("Failed opening sphere discretization file !");
	}

	char fLine[128];
	int nodesRead = 0;
	float px, py, pz;

	for(int n=0; n<numPoints; n++) {
		fscanf(fs, "%f,%f,%f\n", &px, &py, &pz);
		px *= 0.5f*flpDat->ParticleDiam; // Scale points (from unit sphere) by particle radius
		py *= 0.5f*flpDat->ParticleDiam;
		pz *= 0.5f*flpDat->ParticleDiam;
		(*spherePoints)[n] = (cl_float4){px, py, pz, 0.0};
		//printf("Sphere point %d read %f,%f,%f\n", n, px, py, pz);
	}

	fclose(fs);
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
			else if ((inputDefaults+l)->dataType == TYPE_FLOAT_4VEC)
			{
				cl_float value4VecFloat[4];
				sscanf(fLine, "%*s %f %f %f %f", value4VecFloat, value4VecFloat+1, value4VecFloat+2, value4VecFloat+3);
				memcpy( (inputDefaults+l)->varPtr, value4VecFloat, sizeof(cl_float)*4);
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


void compute_shear_stress(host_param_struct* hostDat, int_param_struct* intDat, cl_float* u_h, cl_float* tau_lb_h, int frame)
{
	// Write fluid
	int n_x = intDat->LatticeSize[0];
	int n_y = intDat->LatticeSize[1];
	int n_z = intDat->LatticeSize[2];
	int n_s = hostDat->FluidOutputSpacing; // for float division
	int n_L = n_x*n_y*n_z;

	FILE* fPtr;
	fPtr = fopen ("velocity_profile_z.txt","w");

	int buffer = 0; // Additional buffer to add before starting finite-difference derivatives

	float n_xy = (float)(n_x-2)*(n_y-2);

	float* uMeanX;
	uMeanX = (float*)calloc(n_z,sizeof(float));

	for(int i_z=1; i_z < n_z-1; i_z++) {
		//
		float uMean[3] = {0.0f, 0.0f, 0.0f};
		//
		for(int i_x=1; i_x < n_x-1; i_x++) {
			for(int i_y=1; i_y < n_y-1; i_y++) {

				int i_1D = i_x + intDat->LatticeSize[0]*(i_y + intDat->LatticeSize[1]*i_z);

				uMean[0] += u_h[i_1D];
				uMean[1] += u_h[i_1D + n_L];
				uMean[2] += u_h[i_1D + 2*n_L];

				// Index, then velocity
				//fprintf(vidPtr, "1 "); // Fluid nodes type 1
				//fprintf(vidPtr, "%d %d %d ", i_x-1, i_y-1, i_z-1);
				//fprintf(vidPtr, "%8.6f %8.6f %8.6f\n", u_h[i_1D],  u_h[i_1D + n_L],  u_h[i_1D + 2*n_L]);

			}
		}
		uMean[0] /= n_xy; uMeanX[i_z] = uMean[0];
		uMean[1] /= n_xy;
		uMean[2] /= n_xy;
		fprintf(fPtr, "%8.6f %8.6f %8.6f\n", uMean[0], uMean[1], uMean[2]);

		// Forwards difference from lower boundary
		if (i_z == (2+buffer)) {
			float dudz = uMeanX[i_z]-uMeanX[i_z-1];
			printf("Shear rate (1st order) = %f\n", dudz);
		}
		if (i_z == (3+buffer)) {
			float dudz = 0.5*(uMeanX[i_z]-uMeanX[i_z-2]);
			printf("Shear rate (2nd order) = %f\n", dudz);
		}

	}
	fclose(fPtr);
}


void continuous_output(host_param_struct* hostDat, int_param_struct* intDat, cl_float* u_h, cl_float4* parKin, FILE* vidPtr, int frame)
{
	// Write fluid
	int n_x = intDat->LatticeSize[0];
	int n_y = intDat->LatticeSize[1];
	int n_z = intDat->LatticeSize[2];
	int n_s = hostDat->FluidOutputSpacing; // for float division
	int n_L = n_x*n_y*n_z;

	int n_fluid = (int)(1+(n_x-3)/n_s)*(1+(n_y-3)/n_s)*(1+(n_z-3)/n_s);
	int n_par = intDat->NumParticles;

	fprintf(vidPtr, "%d\n", n_fluid+n_par);
	fprintf(vidPtr, "D3Q19_output, frame %d\n", frame);

	for(int i_x=1; i_x < n_x-1; i_x += n_s) {
		for(int i_y=1; i_y < n_y-1; i_y += n_s) {
			for(int i_z=1; i_z < n_z-1; i_z += n_s) {

				int i_1D = i_x + intDat->LatticeSize[0]*(i_y + intDat->LatticeSize[1]*i_z);

				// Index, then velocity
				fprintf(vidPtr, "1 "); // Fluid nodes type 1
				fprintf(vidPtr, "%d %d %d ", i_x-1, i_y-1, i_z-1);
				fprintf(vidPtr, "%8.6f %8.6f %8.6f\n", u_h[i_1D],  u_h[i_1D + n_L],  u_h[i_1D + 2*n_L]);

			}
		}
	}

	// Write particles
	for (int p = 0; p < n_par; p++) {

		// Index, then velocity
		fprintf(vidPtr, "2 "); // Particles type 2
		fprintf(vidPtr, "%.2f %.2f %.2f ", parKin[p].x, parKin[p].y, parKin[p].z);
		fprintf(vidPtr, "%8.6f %8.6f %8.6f\n", parKin[p+n_par].x, parKin[p+n_par].y, parKin[p+n_par].z);

	}


	// Measure velocity gradient at wall


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
	fclose(fPtr);

	FILE* fPtr2;
	fPtr2 = fopen ("matlab_postproc/velocity_field_centerline.txt","w");

	int i_y = floor((float)intDat->LatticeSize[1]/2.0f);

	for(int i_x=1; i_x < intDat->LatticeSize[0]-1; i_x++) {
		for(int i_z=1; i_z < intDat->LatticeSize[2]-1; i_z++) {

			int i_1D = i_x + intDat->LatticeSize[0]*(i_y + intDat->LatticeSize[1]*i_z);

			// Index, then velocity
			fprintf(fPtr2, "%d %d %d ", i_x, i_y, i_z);
			fprintf(fPtr2, "%9.7f %9.7f %9.7f\n", field[i_1D],  field[i_1D + n_C],  field[i_1D + 2*n_C]);

		}
	}
	fclose(fPtr2);

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
void analyse_platform(cl_device_id* devices, host_param_struct* hostDat)
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
	for(int i=0; i < (int)numCPUs; i++)
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
	cl_uint* numCompUnits = (cl_uint*)calloc(numGPUs, sizeof(cl_uint));
	for(int i=0; i < (int)numGPUs; i++)
	{
		char buf_name[1024];
		cl_uint buf_freq;
		cl_ulong buf_mem;

		printf("GPU info, num. %i\n", i);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_NAME, sizeof(buf_name), buf_name, NULL);
		printf("DEVICE_NAME = %s\n", buf_name);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), numCompUnits+i, NULL);
		printf("DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)numCompUnits[i]);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_freq), &buf_freq, NULL);
		printf("DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_freq);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_mem), &buf_mem, NULL);
		printf("DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_mem);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(buf_mem), &buf_mem, NULL);
		printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = %llu\n", (unsigned long long)buf_mem);

		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), &hostDat->WorkItemSizes, NULL);
		printf("CL_DEVICE_MAX_WORK_ITEM_SIZES = %lu %lu %lu \n", (unsigned long)hostDat->WorkItemSizes[0],
			(unsigned long)hostDat->WorkItemSizes[1], (unsigned long)hostDat->WorkItemSizes[2]);
		clGetDeviceInfo(devicePtrGPU[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &hostDat->MaxWorkGroupSize, NULL);
		printf("CL_DEVICE_MAX_WORK_GROUP_SIZE = %lu \n\n", (unsigned long)hostDat->MaxWorkGroupSize);
	}

	// Choose GPU with most compute units
	cl_uint chosenOne = 0;
	cl_uint chosenUnits = numCompUnits[0];
	for(int i=0; i < (int)numGPUs; i++) {
		if (numCompUnits[i] > chosenUnits) {
			chosenOne = i;
			chosenUnits = numCompUnits[i];
		}
	}
	printf("Choosing device %lu with %lu compute units\n\n", (unsigned long)(chosenOne), (unsigned long)(chosenUnits));

	// Deal with case where there is no GPU...

	// Use default devices for now
	devices[0] = devicePtrCPU[0];
	devices[1] = devicePtrGPU[chosenOne];

	free(platforms);
	free(platformName);
	free(devicePtrCPU);
	free(devicePtrGPU);
}

int create_periodic_stream_mapping(int_param_struct* intDat, cl_int** strMapPtr)
{
	int N_x = intDat->LatticeSize[0];
	int N_y = intDat->LatticeSize[1];
	int N_z = intDat->LatticeSize[2];

	int b = 1; // Buffer layer thickness

	// Constant coord for each face and its value, then upper range of loop for other 2 axis
	// x-axis on inner loop where possible
	int faceSpec[6][6] = {
		{0,b,		2,N_z-b-2, 1,N_y-b-2}, // N-b-2 because edges/verts of face treated separately
		{0,N_x-b-1, 2,N_z-b-2, 1,N_y-b-2}, // Loops are inclusive (<=) upper range
		{1,b,		2,N_z-b-2, 0,N_x-b-2},
		{1,N_y-b-1, 2,N_z-b-2, 0,N_x-b-2},
		{2,b,		1,N_y-b-2, 0,N_x-b-2},
		{2,N_z-b-1, 1,N_y-b-2, 0,N_x-b-2}
	};

	// Two constant coords for each edge and their value, then upper range of the edge axis
	int edgeSpec[12][6] = {
		{0,b,		1,b,	   2,N_z-b-2},
		{0,b,		1,N_y-b-1, 2,N_z-b-2},
		{0,b,		2,b,	   1,N_y-b-2},
		{0,b,		2,N_z-b-1, 1,N_y-b-2},
		{0,N_x-b-1, 1,b,	   2,N_z-b-2},
		{0,N_x-b-1, 1,N_y-b-1, 2,N_z-b-2},
		{0,N_x-b-1, 2,b,	   1,N_y-b-2},
		{0,N_x-b-1, 2,N_z-b-1, 1,N_y-b-2},
		{1,b,		2,b,	   0,N_x-b-2},
		{1,b,		2,N_z-b-1, 0,N_x-b-2},
		{1,N_y-b-1, 2,b,	   0,N_x-b-2},
		{1,N_y-b-1, 2,N_z-b-1, 0,N_x-b-2}
	};

	// Coords of vertexes
	int vertSpec[8][3] = {
		{b,			b,		   b	  },
		{b,			b,		   N_z-b-1},
		{b,			N_y-b-1,   b	  },
		{b,			N_y-b-1,   N_z-b-1},
		{N_x-b-1,	b,		   b	  },
		{N_x-b-1,	b,		   N_z-b-1},
		{N_x-b-1,	N_y-b-1,   b	  },
		{N_x-b-1,	N_y-b-1,   N_z-b-1}
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

					tempMapping[numPeriodicNodes*2	  ] = i_1D; // 1D index of node in f array
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
		||	intDat->BoundaryConds[edgeSpec[edge][2]] == 0) {

			for (int i=b+1; i<=edgeSpec[edge][5]; i++) {

				// Coords of this node
				int r[3];
				r[edgeSpec[edge][0]] = edgeSpec[edge][1]; // Constant coords of edge
				r[edgeSpec[edge][2]] = edgeSpec[edge][3];
				r[edgeSpec[edge][4]] = i;

				// 1D index of node
				int i_1D = r[0] + N_x*(r[1] + r[2]*N_y);

				tempMapping[numPeriodicNodes*2	  ] = i_1D; // 1D index of node f array
				tempMapping[numPeriodicNodes*2 + 1] = 6 + edge; // Boundary node type

				numPeriodicNodes++;
			}
		}
	}

	// Verticies (any boundaries periodic)
	if (intDat->BoundaryConds[0] == 0
	||	intDat->BoundaryConds[1] == 0
	||	intDat->BoundaryConds[2] == 0) {

		for (int vert=0; vert<8; vert++){

			// Coords of this node
			int r[3];
			r[0] = vertSpec[vert][0]; // Constant coords of vert
			r[1] = vertSpec[vert][1];
			r[2] = vertSpec[vert][2];

			// 1D index of node
			int i_1D = r[0] + N_x*(r[1] + r[2]*N_y);

			tempMapping[numPeriodicNodes*2	  ] = i_1D; // 1D index of node in f array
			tempMapping[numPeriodicNodes*2 + 1] = 18 + vert; // Boundary node type

			numPeriodicNodes++;
		}
	}

	// Copy to mapping array with thread-coalesced memory layout
	*strMapPtr = (cl_int*)malloc(numPeriodicNodes*2*sizeof(cl_int));
	for (int node=0; node<numPeriodicNodes; node++) {
		(*strMapPtr)[					node] = tempMapping[node*2	  ]; // 1D index
		(*strMapPtr)[numPeriodicNodes + node] = tempMapping[node*2 + 1]; // Boundary node type
	}

	free(tempMapping);

	return numPeriodicNodes;
}

int equilibrium_distribution_D3Q19(float rho, float* vel, float* f_eq)
{
	float vx = vel[0];
	float vy = vel[1];
	float vz = vel[2];

	float vsq = vx*vx + vy*vy + vz*vz;

	// f_eq = w*rho*(1 + 3*v.c + 4.5*(v.c)^2 - 1.5*|v|^2)

	f_eq[0] = (rho/3.0f)*(1.0f - 1.5f*vsq);

	f_eq[1] = (rho/18.0f)*(1.0f + 3.0f*vx + 4.5f*vx*vx - 1.5f*vsq);
	f_eq[2] = (rho/18.0f)*(1.0f - 3.0f*vx + 4.5f*vx*vx - 1.5f*vsq);
	f_eq[3] = (rho/18.0f)*(1.0f + 3.0f*vy + 4.5f*vy*vy - 1.5f*vsq);
	f_eq[4] = (rho/18.0f)*(1.0f - 3.0f*vy + 4.5f*vy*vy - 1.5f*vsq);
	f_eq[5] = (rho/18.0f)*(1.0f + 3.0f*vz + 4.5f*vz*vz - 1.5f*vsq);
	f_eq[6] = (rho/18.0f)*(1.0f - 3.0f*vz + 4.5f*vz*vz - 1.5f*vsq);

	f_eq[7] = (rho/36.0f)*(1.0f + 3.0f*(vx+vy) + 4.5f*(vx+vy)*(vx+vy) - 1.5f*vsq);
	f_eq[8] = (rho/36.0f)*(1.0f + 3.0f*(vx-vy) + 4.5f*(vx-vy)*(vx-vy) - 1.5f*vsq);
	f_eq[9] = (rho/36.0f)*(1.0f + 3.0f*(vx+vz) + 4.5f*(vx+vz)*(vx+vz) - 1.5f*vsq);
	f_eq[10] = (rho/36.0f)*(1.0f + 3.0f*(vx-vz) + 4.5f*(vx-vz)*(vx-vz) - 1.5f*vsq);

	f_eq[11] = (rho/36.0f)*(1.0f + 3.0f*(-vx+vy) + 4.5f*(-vx+vy)*(-vx+vy) - 1.5f*vsq);
	f_eq[12] = (rho/36.0f)*(1.0f + 3.0f*(-vx-vy) + 4.5f*(-vx-vy)*(-vx-vy) - 1.5f*vsq);
	f_eq[13] = (rho/36.0f)*(1.0f + 3.0f*(-vx+vz) + 4.5f*(-vx+vz)*(-vx+vz) - 1.5f*vsq);
	f_eq[14] = (rho/36.0f)*(1.0f + 3.0f*(-vx-vz) + 4.5f*(-vx-vz)*(-vx-vz) - 1.5f*vsq);

	f_eq[15] = (rho/36.0f)*(1.0f + 3.0f*(vy+vz) + 4.5f*(vy+vz)*(vy+vz) - 1.5f*vsq);
	f_eq[16] = (rho/36.0f)*(1.0f + 3.0f*(vy-vz) + 4.5f*(vy-vz)*(vy-vz) - 1.5f*vsq);
	f_eq[17] = (rho/36.0f)*(1.0f + 3.0f*(-vy+vz) + 4.5f*(-vy+vz)*(-vy+vz) - 1.5f*vsq);
	f_eq[18] = (rho/36.0f)*(1.0f + 3.0f*(-vy-vz) + 4.5f*(-vy-vz)*(-vy-vz) - 1.5f*vsq);

	return 0;
}

float compute_tau(int viscosityModel, float srtII, cl_float NewtonianTau, cl_float* nonNewtonianParams)
{
	float tau;

	if (viscosityModel == VISC_NEWTONIAN) {
		tau = NewtonianTau;
	}
	else if (viscosityModel == VISC_POWER_LAW) {

		float k = nonNewtonianParams[0];
		float n = nonNewtonianParams[1];
		float nu;

		// Faster to check for common values than to use pow() always
		if (n == 0.5f) {
			nu = k/sqrt(srtII);
		}
		else if (n == 1.0f) {
			nu = k;
		}
		else if (n == 2.0f) {
			nu = k*srtII;
		}
		else {
			nu = k*pow(srtII,n-1.0f);
		}

		tau = 3.0f*nu + 0.5f;
	}
	else if (viscosityModel == VISC_CASSON) {

		float tau_Y = nonNewtonianParams[0];
		float eta_inf = nonNewtonianParams[1];

		float nu = (sqrt(tau_Y/srtII) + sqrt(eta_inf))*(sqrt(tau_Y/srtII) + sqrt(eta_inf));
		tau = 3.0f*nu + 0.5f;

	}
	else if (viscosityModel == VISC_HB) {
		float tau_Y = nonNewtonianParams[0];
		float k = nonNewtonianParams[1];
		float n = nonNewtonianParams[2];
		float nu;

		if (n == 0.5f) {
			nu = tau_Y/srtII + k/sqrt(srtII);
		}
		else if (n == 1.0f) {
			nu = tau_Y/srtII + k;
		}
		else if (n == 2.0f) {
			nu = tau_Y/srtII + k*srtII;
		}
		else {
			nu = tau_Y/srtII + k*pow(srtII,n-1.0f);
		}

		tau = 3.0f*nu + 0.5f;
	}

	return tau;
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
		printf("Call to %s >> FAILED << (%d):\n", clFunc, err);

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
	//	  printf("%i %i %i \n", A[i], B[i], C[i]);
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
