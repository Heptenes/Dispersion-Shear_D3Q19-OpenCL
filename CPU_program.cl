
#include "struct_header_device.h"

__kernel void particle_dynamics(
	__global int_param_struct* intDat, 
	__global float4* parKinematics_h,
	__global float4* parForces_h,
	__global zone_struct* zoneDat)
{
	//int thrID = get_global_id(0);
	
	
}


__kernel void update_particle_zones(
	__global int_param_struct* intDat,
	__global float4* parKinematics_h,
	__global float4* parForces_h,
	__global zone_struct* zoneDat)
{
	//int thrID = get_global_id(0);
	
	// Loop over particles for this thread
	
	
	// Determine zone
	
	
	// Add to zone, atomic increment particle counter (must be reset)
	
}
