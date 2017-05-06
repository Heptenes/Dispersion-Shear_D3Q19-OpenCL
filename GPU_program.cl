__kernel void GPU_fluid_collide_stream_D3Q19_SRT(__global int* f, __global float* bf, 
	__constant int_param_struct* intDat, __constant float_param_struct* flDat)
{
	// Get 1D index
	int i1 = get_global_id(0);
	
	// Single relaxtion time (BGK) collision
	//
	

}

__kernel void GPU_fluid_collide_stream_D3Q19_MRT()
{


}

__kernel void GPU_compute_macro_properties(__global int* f)
{
	
}

__kernel void GPU_compute_macro_derivatives(__global int* f)
{
	
}