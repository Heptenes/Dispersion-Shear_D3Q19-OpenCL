#define PAR_COL_HARMONIC 1
#define PAR_COL_LJ 2

typedef struct {

	cl_int MaxIterations;
	
	cl_int BasisVel[19][3];
	cl_int LatticeSize[3];
	cl_int BufferSize[3];
	cl_int BoundaryConds[3];
	cl_int NumZones[3];
	
	cl_int MaintainShear;
	cl_int ViscosityModel;

	cl_int NumParticles;
	cl_int ParForceModel;
	
	cl_int MaxSurfPointsPerNode;
	cl_int InterpOrderIBM;
	
	cl_int PointsPerParticle;
	cl_int PointsPerWorkGroup;
	cl_int TotalSurfPoints;
	cl_int NumForceArrays;
	
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
	
	cl_float ZoneWidth[3];

	// Particle
	cl_float ParticleDiam;
	cl_float ParticleMass;
	cl_float PointArea;
	cl_float ParticleMomInertia;
	cl_float ParForceParams[2];
	
	cl_float DirectForcingCoeff;

} flp_param_struct;

typedef struct {
	cl_int NumNeighbors;
	cl_int NeighborZones[26];
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
