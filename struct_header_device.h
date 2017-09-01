typedef struct {

	int BasisVel[19][3];
	int MaxIterations;
	int LatticeSize[3];
	int BufferSize[3];
	int BoundaryConds[3];
	int MaintainShear;
	int ViscosityModel;

	int NumParticles;
	int MaxSurfPointsPerNode;
	int InterpOrderIBM;
	int TotalSurfPoints;
	int RebuildFreq;

} int_param_struct;


typedef struct {

	// Fluid
	float ConstBodyForce[3];
	float EqWeights[19];
	float VelUpper[3];
	float VelLower[3];
	// Viscosity
	float NewtonianTau;
	float ViscosityParams[4];

	// Particle
	float ParticleSize;

} flp_param_struct;

typedef struct {
	int numParInZone;
} zone_struct;

/*typedef struct {

	// Translational DOF
	float xyz[3];
	float force[3];
	float velocity[3];

	// Rotational DOF
	float rot[3]
	float torque[3];
	float angVel[3];

	// Discretization
	int numNodes;
	float* particleNodes;

} host_particle_struct;*/
