
#include "struct_header_device.h"


__kernel void particle_particle_forces(
	__global int_param_struct* intDat, 
	__global flp_param_struct* flpDat,
	__global float4* parKin,
	__global float4* parForce,
	__global zone_struct* zoneDat,
	__global int* parsZone,
	__global int* parInThread,
	__global uint* numParInThread,
	__global int* parInZone,
	__global uint* numParInZone)
{
	int thrID = get_global_id(0);
	int np = intDat->NumParticles;
	
	for(uint i = 0; i < numParInThread[thrID]; i++)
	{		
		// Detect collisions
		int pi = parInThread[i];
		
		int pZone = parsZone[pi];
		float4 ppForce = (float4){0.0, 0.0, 0.0, 0.0};
		
		// Loop over neighbour zones (which should include this particles zone as well)
		for (int i_nz = 0; i_nz < zoneDat[pZone].NumNeighbors; i_nz++) {
			
			int zoneID = zoneDat->NeighborZones[i_nz];
			
			for (int j = 0; j < numParInZone[zoneID]; j++) {
				
				int pj = parInZone[j];
				
				if (pi > pj) {
					// Distance
					
					float4 rij = parKin[pj] - parKin[pi];
					float rSep = length(rij);
					float4 eij = rij/rSep; // Normalized vector pointing from i to j
					
					float overlap = rSep - flpDat->ParticleDiam;
					
					if (intDat->ParForceModel == PAR_COL_HARMONIC) { // Harmonic
						float fMag = flpDat->ParForceParams[0]*overlap*overlap;
						
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
	__global int* parInThread,
	__global uint* numParInThread)
{
	int thrID = get_global_id(0);
	int np = intDat->NumParticles;
	
	for(uint i = 0; i < numParInThread[thrID]; ++i)
	{
		int p = parInThread[i];
		
		float4 accel = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		float4 angAccel = (float4){0.0f, 0.0f, 0.0f, 0.0f};
		
		// Sum fluid-particle forces
		for (int fa = 0; fa < intDat->NumForceArrays; fa++) {
			accel += parFluidForce[p + np*2*fa];
			angAccel += parFluidForce[p + np*(2*fa+1)];
		}
		
		// Add particle-particle force
		accel += parForce[p];
		angAccel += parForce[p + np];
		
		accel /= flpDat->ParticleMass;
		angAccel /= flpDat->ParticleMomInertia; // Spheres
				
		// Update position (2nd order)
		// x_t+1  =  x_t      +  v_t*dt        + 0.5*acc*dt^2
		parKin[p] = parKin[p] + parKin[p + np] + 0.5f*accel;
		
		// Update velocity
		parKin[p + np] = parKin[p + np] + accel;
		
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
	__global int* parInThread,
	__global uint* numParInThread,
	__global int* parsZone,
	__global int* parInZone,
	__global uint* numParInZone)
{
	int thrID = get_global_id(0);
	
	printf("Thread ID %d", thrID);
	
	// Loop over particles for this thread
	for(uint i = 0; i < numParInThread[thrID]; ++i)
	{
		int p = parInThread[i];
		
		// Particles always belong to their initial thread
		int zoneIDx = parKin[p].x/flpDat->ZoneWidth[0]; // Need to use more vector types
		int zoneIDy = parKin[p].y/flpDat->ZoneWidth[1];
		int zoneIDz = parKin[p].z/flpDat->ZoneWidth[2];
		int zoneID = zoneIDx + intDat->NumZones[0]*(zoneIDy + intDat->NumZones[2]*zoneIDz);
		
		parsZone[p] = zoneID;
		parInZone[zoneID*intDat->NumParticles + numParInZone[zoneID]++] = p;
	}
	
	// Add to zone, atomic increment particle counter (must be reset)
	
}
