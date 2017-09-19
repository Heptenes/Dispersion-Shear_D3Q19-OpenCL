#define PAR_COL_HARMONIC 1
#define PAR_COL_LJ 2

typedef struct {

	int MaxIterations;
	
	int BasisVel[19][3];
	
	int LatticeSize[3];
	int SystemSize[3];
	
	int BufferSize[3];
	int BoundaryConds[3];
	int NumZones[3];
	
	int MaintainShear;
	int ViscosityModel;

	int NumParticles;
	int ParForceModel;
	
	int MaxSurfPointsPerNode;
	int InterpOrderIBM;
	
	int PointsPerParticle;
	int PointsPerWorkGroup;
	int TotalSurfPoints;
	int WorkGroupsPerParticle;

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
	
	float ZoneWidth[3];

	// Particle
	float ParticleDiam;
	float ParticleMass;
	float PointArea;
	float ParticleMomInertia;
	float ParForceParams[2];
	float ParticleZBuffer;
	
	float DirectForcingCoeff;

} flp_param_struct;

//typedef struct {
//	int NeighborZones[32];
//	int NumNeighbors;
//}  zone_struct;

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
