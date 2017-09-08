
//#include "struct_header_device.h"

#define PAR_COL_HARMONIC 1
#define PAR_COL_LJ 2

typedef struct {

	int MaxIterations;

	int BasisVel[19][3];
	int LatticeSize[3];
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
	int NumForceArrays;

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

	float ZoneWidth[3];

	// Particle
	float ParticleDiam;
	float ParticleMass;
	float PointArea;
	float ParticleMomInertia;
	float ParForceParams[2];

	float DirectForcingCoeff;

} flp_param_struct;

typedef struct {
	int NumNeighbors;
	int NeighborZones[26];
} zone_struct;

__kernel void particle_particle_forces(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global float4* parForce,
	__global zone_struct* zoneDat,
	__global int* parsZone,
	__global int* threadMembers,
	__global uint* numParInThread,
	__global int* zoneMembers,
	__global uint* numParInZone)
{
	int threadID = get_global_id(0);

	for(uint i = 0; i < numParInThread[threadID]; i++)
	{
		// Detect collisions
		int pi = threadMembers[threadID*intDat->NumParticles + i];

		int pZone = parsZone[pi];
		//printf("zoneDat[pZone].NumNeighbors = %d\n", zoneDat[pZone].NumNeighbors);

		// Loop over neighbour zones (which should include this particles zone as well)
		for (int i_nz = 0; i_nz < zoneDat[pZone].NumNeighbors; i_nz++) {

			int zoneID = zoneDat->NeighborZones[i_nz];

			for (int j = 0; j < numParInZone[zoneID]; j++) {

				int pj = zoneMembers[zoneID*intDat->NumParticles + j];

				if (pi > pj) {
					//printf("Testing for collision between particles %d and %d\n", pi, pj);
					// Distance
					float4 rij = parKin[pj] - parKin[pi];
					float rSep = length(rij);
					float4 eij = rij/rSep; // Normalized vector pointing from i to j

					//printf("rij, |r| = (%f,%f,%f) %f\n", rij.x, rij.y, rij.z, rSep);

					float overlap = rSep - flpDat->ParticleDiam;

					if (intDat->ParForceModel == PAR_COL_HARMONIC) { // Harmonic f = k.x
						float fMag = flpDat->ParForceParams[0]*overlap;

						// Update forces (no torque contribution)
						parForce[pi] -= eij*fMag;
						parForce[pj] += eij*fMag; // Need to make safe against race conditions
					}
				}
			}
		}
	}
}


__kernel void particle_dynamics(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global float4* parForce,
	__global float4* parFluidForce,
	__global zone_struct* zoneDat,
	__global int* parsZone,
	__global int* threadMembers,
	__global uint* numParInThread)
{
	int threadID = get_global_id(0);
	int np = intDat->NumParticles;
	int nfa = intDat->NumForceArrays;

	for(uint i = 0; i < numParInThread[threadID]; ++i)
	{
		int p = threadMembers[i];

		float4 accel = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		float4 angAccel = (float4){0.0f, 0.0f, 0.0f, 0.0f};

		// Sum fluid-particle forces
		for (int fa = 0; fa < nfa; fa++) {
			accel += parFluidForce[p*nfa + fa]; // Force
			printf("force = %f %f %f\n", accel.x, accel.y, accel.z);
			angAccel += parFluidForce[(p+np)*nfa + fa]; // Torque
			printf("torque = %f %f %f\n", angAccel.x, angAccel.y, angAccel.z);
		}

		// Add particle-particle force
		accel += parForce[p];
		angAccel += parForce[p + np];

		accel /= flpDat->ParticleMass;
		angAccel /= flpDat->ParticleMomInertia; // Spheres

		// Update position (2nd order)
		// x_t+1  =  x_t      +  v_t*dt        + 0.5*acc*dt^2
		printf("Old position: %f %f %f\n", parKin[p].x, parKin[p].y, parKin[p].z);
		parKin[p] = parKin[p] + parKin[p + np] + 0.5f*accel;
		printf("New position: %f %f %f\n", parKin[p].x, parKin[p].y, parKin[p].z);

		// Update velocity
		//printf("Old velocity: %f %f %f\n", parKin[p+np].x, parKin[p+np].y, parKin[p+np].z);
		parKin[p + np] = parKin[p + np] + accel;
		//printf("New velocity: %f %f %f\n\n", parKin[p+np].x, parKin[p+np].y, parKin[p+np].z);

		// Update rotation quaternion
		// dq/dt = (1/2)*[angVel, 0]*q where [angVel, 0] and q are quaternions
		float4 q = parKin[p + 2*np];          // Current quaternion
		float4 av = parKin[p + 3*np];         // Angular velocity

		float dq_w = -dot(av,q);              // Scalar part of dq/dt
		float4 dq_xyz = q.w*av + cross(av,q); // Vector part

		parKin[p + 2*np] = 0.5f*dq_xyz;
		parKin[p + 2*np].w += 0.5f*dq_w;

		// Update angular velocity, dOmega = dt*T/I
		parKin[p + 3*np] = av + angAccel;
	}
}


__kernel void update_particle_zones(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global zone_struct* zoneDat,
	__global int* threadMembers,
	__global uint* numParInThread,
	__global int* parsZone,
	__global int* zoneMembers,
	__global uint* numParInZone)
{
	int threadID = get_global_id(0);

	//printf("update_particle_zones: Thread ID %d,", threadID);
	//printf(" num par in thread = %d\n", numParInThread[threadID]);

	// Loop over particles for this thread
	for(uint i = 0; i < numParInThread[threadID]; ++i)
	{
		int p = threadMembers[i + threadID*intDat->NumParticles];

		// Particles always belong to their initial thread
		int zoneIDx = parKin[p].x/flpDat->ZoneWidth[0]; // Need to use more vector types
		int zoneIDy = parKin[p].y/flpDat->ZoneWidth[1];
		int zoneIDz = parKin[p].z/flpDat->ZoneWidth[2];
		int zoneID = zoneIDx + intDat->NumZones[0]*(zoneIDy + intDat->NumZones[2]*zoneIDz);

		parsZone[p] = zoneID;
		zoneMembers[zoneID*intDat->NumParticles + numParInZone[zoneID]++] = p;

		//printf("Particle %d now in zone %d\n", p, zoneID);
	}

	// Add to zone, atomic increment particle counter (must be reset)

}
