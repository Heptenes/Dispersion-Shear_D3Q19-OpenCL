
//#include "struct_header_device.h"

#define PAR_COL_HARMONIC 1
#define PAR_COL_LJ 2

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#define VISC_NEWTONIAN 1
#define VISC_POWER_LAW 2
#define VISC_HB 3
#define VISC_CASSON 4

#define MIN_SEP 0.1
#define SQUEEZE_RANGE 0.1

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
	int NumForceArrays;

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

float compute_squeeze_force(int viscosityModel, float vRel, float minSep, float rp, float NewtonianTau, __global float* nonNewtonianParams);

__kernel void particle_particle_forces(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global float4* parForce,
	__global int* zoneNeighDat,
	__global int* parsZone,
	__global int* threadMembers,
	__global int* numParInThread,
	__global int* zoneMembers,
	__global int* numParInZone)
{
	int threadID = get_global_id(0);
	int np = intDat->NumParticles;
	float rp = flpDat->ParticleDiam/2.0f;
	
	int N_x = intDat->LatticeSize[0];
	int N_y = intDat->LatticeSize[1];
	int N_z = intDat->LatticeSize[2];
	float4 w = (float4)(intDat->SystemSize[0], intDat->SystemSize[1], intDat->SystemSize[2], 1.0f); 

	for(int i = 0; i < numParInThread[threadID]; i++)
	{
		// Detect collisions
		int pi = threadMembers[threadID*intDat->NumParticles + i];
		//printf("pi = %d\n", pi);

		int pZone = parsZone[pi];
		//printf("zoneNeighDat[pZone*28] = %d\n", zoneNeighDat[pZone*28]);

		// Loop over neighbour zones (which should include this particles zone as well)
		for (int i_nz = 1; i_nz <= zoneNeighDat[pZone*28]; i_nz++) {

			int zoneID = zoneNeighDat[pZone*28 + i_nz];
			//printf("zoneID = %d\n", zoneID);
			//printf("numParInZone[zoneID] = %d\n", numParInZone[zoneID]);

			for (int j = 0; j < numParInZone[zoneID]; j++) {

				int pj = zoneMembers[zoneID*np + j];
				//printf("pj = %d\n", pj);

				if (pi > pj) {
					//printf("Testing for collision between particles %d and %d\n", pi, pj);
					
					// Distance
					float4 rij = parKin[pj] - parKin[pi];
					// Correct each component for pbc 
					int signX = (rij.x < 0.0f) ? -1 : (rij.x > 0.0f);
					int signY = (rij.y < 0.0f) ? -1 : (rij.y > 0.0f);
					int signZ = (rij.z < 0.0f) ? -1 : (rij.z > 0.0f);
					//printf("    rij, |r| = (%f,%f,%f)\n", rij.x, rij.y, rij.z);
					rij.x = (fabs(rij.x) < w.x/2.0f) ? rij.x : (rij.x - (float)signX*w.x);
					rij.y = (fabs(rij.y) < w.y/2.0f) ? rij.y : (rij.y - (float)signY*w.y);
					rij.z = (fabs(rij.z) < w.z/2.0f) ? rij.z : (rij.z - (float)signZ*w.z);
					//printf("pbc rij, |r| = (%f,%f,%f)\n", rij.x, rij.y, rij.z);
					
					float rSep = length(rij); // 4th component needs to be zero, which it should be
					
					// Relative velocity
					float4 vij = parKin[pj + np] - parKin[pi + np];
					
					float minSep = rSep - 2*rp;	// Closest approach between spheres
					float4 eij = rij/rSep;
					float vRel = -(eij.x*vij.x + eij.y*vij.y + eij.z*vij.z); // Positive if spheres approaching
					//printf("vRel = (%f,%f,%f) %f\n", vij.x, vij.y, vij.z, vRel);

					float overlap = -minSep;
					//printf("overlap = %f\n", overlap);

					if (overlap > 0 && intDat->ParForceModel == PAR_COL_HARMONIC) { // Harmonic f = k.x
						//printf("Harmonic collision between particles %d and %d\n", pi, pj);
						
						float fMag = flpDat->ParForceParams[0]*overlap;
						//printf("fMag = %f\n", fMag);

						// Update forces (no torque contribution)
						parForce[pi] -= eij*fMag;
						parForce[pj] += eij*fMag; // Need to make safe against race conditions
					}
					
					
					// Squeeze force
					if (minSep > 0 && minSep < SQUEEZE_RANGE*rp) { // Harmonic f = k.x
						
						float sqForce = compute_squeeze_force(intDat->ViscosityModel, vRel, minSep, rp, 
							flpDat->NewtonianTau, &(flpDat->ViscosityParams[0]));
							
						//printf("Squeeze force = %f\n", sqForce);
							
						parForce[pi] -= eij*sqForce;
						parForce[pj] += eij*sqForce;
						
					}
				}
			}
		}
		
		// Particle-wall collisions (z-wall only)
		float lowerOverlap = flpDat->ParticleZBuffer - parKin[pi].z;
		if (lowerOverlap > 0) {
			parForce[pi].z += flpDat->ParForceParams[0]*lowerOverlap;
		}
		float upperOverlap = parKin[pi].z - ((float)intDat->SystemSize[2]-flpDat->ParticleZBuffer);
		if (upperOverlap > 0) {
			parForce[pi].z -= flpDat->ParForceParams[0]*upperOverlap;
		}
	}
}


__kernel void particle_dynamics(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global float4* parForce,
	__global float4* parFluidForce,
	__global int* threadMembers,
	__global int* numParInThread)
{
	int threadID = get_global_id(0);
	int np = intDat->NumParticles;
	int nfa = intDat->NumForceArrays;
	
	//printf("ThreadID = %d\n", threadID);
	//printf("numParInThread[threadID] = %d\n", numParInThread[threadID]);
	//printf("threadMembers_h[%d*intDat.NumParticles] = %d\n", threadID, threadMembers[threadID*intDat->NumParticles]);
		
	int N_x = intDat->LatticeSize[0];
	int N_y = intDat->LatticeSize[1];
	int N_z = intDat->LatticeSize[2];
	float4 w = (float4)(intDat->SystemSize[0], intDat->SystemSize[1], intDat->SystemSize[2], 1.0f); 

	for(int i = 0; i < numParInThread[threadID]; ++i)
	{
		int p = threadMembers[threadID*intDat->NumParticles + i];
		//printf("Updating particle %d\n", p);

		float4 accel = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		float4 angAccel = (float4){0.0f, 0.0f, 0.0f, 0.0f};

		// Sum fluid-particle forces
		for (int fa = 0; fa < nfa; fa++) {
			accel += parFluidForce[p*nfa + fa]; // Force
			//printf("Fluid-particle force    =  %f %f %f\n", accel.x, accel.y, accel.z);
			angAccel += parFluidForce[(p+np)*nfa + fa]; // Torque
			//printf("Fluid-particle torque   = %f %f %f\n", angAccel.x, angAccel.y, angAccel.z);
		}

		// Add particle-particle force
		//printf("Particle-particle force = %f %f %f\n", parForce[p].x, parForce[p].y, parForce[p].z);
		accel += parForce[p];
		angAccel += parForce[p + np];
		//printf("Particle-particle force = %f %f %f\n", parForce[p].x, parForce[p].y, parForce[p].z);
		//printf("+ particle-particle torque   = %f %f %f\n", angAccel.x, angAccel.y, angAccel.z);

		accel /= flpDat->ParticleMass;
		angAccel /= flpDat->ParticleMomInertia; // Spheres
		//printf("ang accel   = %f %f %f\n", angAccel.x, angAccel.y, angAccel.z);

		// Add constant acceleration
		accel.x += flpDat->ConstBodyForce[0]; // Need to move to vector implementation
		accel.y += flpDat->ConstBodyForce[1];
		accel.z += flpDat->ConstBodyForce[2];
		//printf("Total acceleration      =   %f %f %f\n", accel.x, accel.y, accel.z);


		// Update position (2nd order)
		// x_t+1  =  x_t      +  v_t*dt        + 0.5*acc*dt^2
		//printf("Old position: %f %f %f\n", parKin[p].x, parKin[p].y, parKin[p].z);
		float4 rTemp = parKin[p] + parKin[p + np] + 0.5f*accel;
		parKin[p] = fmod((rTemp+w),w);
		//printf("New position: %f %f %f\n", parKin[p].x, parKin[p].y, parKin[p].z);


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
		//printf("Old ang. vel: %f %f %f\n", av.x, av.y, av.z);
		parKin[p + 3*np] = av + angAccel;
		//printf("New ang. vel: %f %f %f\n", parKin[p+3*np].x, parKin[p+3*np].y, parKin[p+3*np].z);
		
		// Reset force and torque
		parForce[p] = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		parForce[p + np] = (float4){0.0f, 0.0f, 0.0f, 0.0f};
	}
}


__kernel void update_particle_zones(
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global int* threadMembers,
	__global int* numParInThread,
	__global int* parsZone,
	__global int* zoneMembers,
	__global int* numParInZone)
{
	int threadID = get_global_id(0);

	//printf("update_particle_zones: Thread ID %d,", threadID);
	//printf(" num par in thread = %d\n", numParInThread[threadID]);

	// Loop over particles for this thread
	for(int i = 0; i < numParInThread[threadID]; ++i)
	{
		int p = threadMembers[i + threadID*intDat->NumParticles];

		// Particles always belong to their initial thread
		int zoneIDx = (int)(parKin[p].x/flpDat->ZoneWidth[0]);
		int zoneIDy = (int)(parKin[p].y/flpDat->ZoneWidth[1]);
		int zoneIDz = (int)(parKin[p].z/flpDat->ZoneWidth[2]);
		int zoneID = zoneIDx + intDat->NumZones[0]*(zoneIDy + intDat->NumZones[2]*zoneIDz);
		//printf("ZoneID x,y,z = %d,%d,%d", zoneIDx, zoneIDy, zoneIDz);

		parsZone[p] = zoneID;
		zoneMembers[zoneID*intDat->NumParticles + numParInZone[zoneID]++] = p;

		//printf("Particle %d now in zone %d\n", p, zoneID);
	}

	// Add to zone, atomic increment particle counter (must be reset)

}



float compute_squeeze_force(int viscosityModel, float vRel, float minSep, float rp, float NewtonianTau, __global float* nonNewtonianParams)
{
	float rStar = rp/2.0f;
	minSep = minSep < MIN_SEP ? MIN_SEP : minSep;
	float force = 0.0;
	
	if (viscosityModel == VISC_NEWTONIAN) {
		
		float eta0 = (NewtonianTau-0.5f)/3.0f;
		force = 6*M_PI*eta0*rStar*rStar*vRel/minSep;
	}
	else if (viscosityModel == VISC_POWER_LAW) {


	}
	else if (viscosityModel == VISC_CASSON) {


	}
	else if (viscosityModel == VISC_HB) {
		
	}

	return force;
}
