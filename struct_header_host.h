typedef struct {

	cl_int BasisVel[19][3];
	cl_int MaxIterations;
	cl_int LatticeSize[3];
	cl_int BufferSize[3];
	cl_int BoundaryConds[3];
	cl_int MaintainShear;
	cl_int ViscosityModel;

	cl_int NumParticles;
	cl_int MaxSurfPointsPerNode;
	cl_int InterpOrderIBM;
	cl_int TotalSurfPoints;
	cl_int RebuildFreq;

} int_param_struct;


typedef struct {

	// Fluid
	cl_float ConstBodyForce[3];
	cl_float EqWeights[19];
	cl_float VelUpper[3];
	cl_float VelLower[3];
	// Viscosity
	cl_float NewtonianTau;
	cl_float ViscosityParams[4];

	// Particle
	cl_float ParticleSize;

} flp_param_struct;

typedef struct {
	cl_int numParInZone;
} zone_struct;

/*typedef struct {

	// Translational DOF
	cl_float xyz[3];
	cl_float force[3];
	cl_float velocity[3];

	// Rotational DOF
	cl_float rot[3]
	cl_float torque[3];
	cl_float angVel[3];

	// Discretization
	cl_int numNodes;
	cl_float* particleNodes;

} host_particle_struct;*/
