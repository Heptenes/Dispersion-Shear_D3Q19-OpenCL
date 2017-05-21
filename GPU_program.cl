#include "struct_header_device.h"

void equilibirum_distribution_D3Q19(float* f_eq, float rho, float u_x, float u_y, float u_z);
void stream_locations(int n_x, int n_y, int n_z, int i_x, int i_y, int i_z, int* ind);
void guo_body_force_term(float u_x, float u_y, float u_z,
	float g_x, float g_y, float g_z, float* fGuo);

__kernel void GPU_collideSRT_stream_D3Q19(
	__global float* f_c, 
	__global float* f_s,
	__global float* g, 
	__global float* u, 
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat)
{
	// Get 3D indices
	int i_x = get_global_id(0); // Using global_work_offset to take buffer layer into account
	int i_y = get_global_id(1);
	int i_z = get_global_id(2);
	
	// Convention: upper-case N for total lattice array size (including buffer)
	int N_x = intDat->LatticeSize[0]; 
	int N_y = intDat->LatticeSize[1]; 
	int N_z = intDat->LatticeSize[2];
		
	// 1D index
	int i_1D = i_x + N_x*(i_y + N_y*i_z);
	int N_C = N_x*N_y*N_z; // Total nodes
	
	// Read in f (from f_c) for this cell
	float f[19]; 
	
	// Read f_c from __global to private memory (should be coalesced memory access)
	float rho = 0.0f;
	for (int i=0; i<19; i++) {
		f[i] = f_c[ i_1D + i*N_C ];
		rho += f[i];
	}
	
#ifdef USE_CONSTANT_BODY_FORCE
	float g_x = flpDat->ConstBodyForce[0];
	float g_y = flpDat->ConstBodyForce[1];
	float g_z = flpDat->ConstBodyForce[2];
#else	
	float g_x = g[ i_1D         ];
	float g_y = g[ i_1D +   N_C ];
	float g_z = g[ i_1D + 2*N_C ];
#endif
		
	// Compute velocity	(J. Stat. Mech. (2010) P01018 convention)
	// w/ body force contribution (Guo et al. 2002)
	float u_x = (f[1]-f[2]+f[7]+f[8] +f[9] +f[10]-f[11]-f[12]-f[13]-f[14] + 0.5f*g_x)/rho;
	float u_y = (f[3]-f[4]+f[7]-f[8] +f[11]-f[12]+f[15]+f[16]-f[17]-f[18] + 0.5f*g_y)/rho;
	float u_z = (f[5]-f[6]+f[9]-f[10]+f[13]-f[14]+f[15]-f[16]+f[17]-f[18] + 0.5f*g_z)/rho;
		
	// Write to __global *u
	u[ i_1D         ] = u_x;
	u[ i_1D +   N_C ] = u_y;
	u[ i_1D + 2*N_C ] = u_z;
	
	// Single relaxtion time (BGK) collision
	float f_eq[19];
	equilibirum_distribution_D3Q19(f_eq, rho, u_x, u_y, u_z);
	
	//for(size_t i = 0; i < 19; ++i) {
	//	printf("eq dist: %d %e\n", i, f_eq[i]);
	//}
	
	int streamIndex[19];
	stream_locations(N_x, N_y, N_z, i_x, i_y, i_z, streamIndex);
	
	//for(size_t i = 0; i < 19; ++i) {
	//	printf("stream loc: %d %d\n", i, streamIndex[i]);
	//}
			
	// Guo, Zheng & Shi body force term (2002)
	float tau = flpDat->NewtonianTau;
	float fGuo[19];
	guo_body_force_term(u_x, u_y, u_z, g_x, g_y, g_z, fGuo);
	
	float pfGuo = (1.0f - 0.5f/tau); // Guo term SRT collision prefactor 
	
	//float guoSum = 0.0;
	for(int i = 0; i < 19; i++) {
		fGuo[i] *= pfGuo; // SRT relaxation
		//guoSum += fGuo[i];
		//printf("Guo, guosum %d %e %e\n", i, fGuo[i], guoSum);
	}
	
	// Propagate to f_s (not taking into account boundary conditions)
	for (int i=0; i<19; i++) {
		f_s[streamIndex[i]] = f[i] + (f_eq[i]-f[i])/tau + fGuo[i];
	}
	
}

__kernel void GPU_boundary_periodic(__global float* f_s,
	__global int_param_struct* intDat,
	__global int* streamMapping)
{
	int i_k = get_global_id(0);
	int N_BC = get_global_size(0); // Total number of periodic boundary nodes
	
	int i_1D = streamMapping[i_k];
	int typeBC = streamMapping[i_k + N_BC];
	
	// Uppercase N to denote total lattice size (including buffer layer)
	int N[3];
	N[0] = intDat->LatticeSize[0];
	N[1] = intDat->LatticeSize[1];
	N[2] = intDat->LatticeSize[2];
	int N_C = N[0]*N[1]*N[2];
		
	// D3Q19 version
	
	// Indexes of unknown f for each type of boundary node
	// First col specifies amount of unknows 
	int unknowns[26][13] = {
	// 6 faces, each 5 unknown components
		{ 5, 1, 7, 8, 9,10, 0, 0, 0, 0, 0, 0, 0}, // x- 0
		{ 5, 2,11,12,13,14, 0, 0, 0, 0, 0, 0, 0}, // x+ 1
		{ 5, 3, 7,11,15,16, 0, 0, 0, 0, 0, 0, 0}, // y- 2
		{ 5, 4, 8,12,17,18, 0, 0, 0, 0, 0, 0, 0}, // y+ 3
		{ 5, 5, 9,13,15,17, 0, 0, 0, 0, 0, 0, 0}, // z- 4
		{ 5, 6,10,14,16,18, 0, 0, 0, 0, 0, 0, 0}, // z+ 5
	// 12 edges, each 9 unknown components
		{ 9, 1, 3, 7, 8, 9,10,11,15,16, 0, 0, 0}, // x-y- 6
		{ 9, 1, 4, 7, 8, 9,10,12,17,18, 0, 0, 0}, // x-y+ 7
		{ 9, 1, 5, 7, 8, 9,10,13,15,17, 0, 0, 0}, // x-z- 8
		{ 9, 1, 6, 7, 8, 9,10,14,16,18, 0, 0, 0}, // x-z+ 9
		{ 9, 2, 3, 7,11,12,13,14,15,16, 0, 0, 0}, // x+y- 10
		{ 9, 2, 4, 8,11,12,13,14,17,18, 0, 0, 0}, // x+y+ 11
		{ 9, 2, 5, 9,11,12,13,14,15,17, 0, 0, 0}, // x+z- 12
		{ 9, 2, 6,10,11,12,13,14,16,18, 0, 0, 0}, // x+z+ 13
		{ 9, 3, 5, 7, 9,11,13,15,16,17, 0, 0, 0}, // y-z- 14
		{ 9, 3, 6, 7,10,11,14,15,16,18, 0, 0, 0}, // y-z+ 15
		{ 9, 4, 5, 8, 9,12,13,15,17,18, 0, 0, 0}, // y+z- 16
		{ 9, 4, 6, 8,10,12,14,16,17,18, 0, 0, 0}, // y+z+ 17
	// 8 vertices, each 12 unknown components
		{ 12, 1, 3, 5, 7, 8, 9,10,11,13,15,16,17}, // x-y-z- 18
		{ 12, 1, 3, 6, 7, 8, 9,10,11,14,15,16,18}, // x-y-z+ 19
		{ 12, 1, 4, 5, 7, 8, 9,10,12,13,15,17,18}, // x-y+z- 20
		{ 12, 1, 4, 6, 7, 8, 9,10,12,14,16,17,18}, // x-y+z+ 21
		{ 12, 2, 3, 5, 7, 9,11,12,13,14,15,16,17}, // x+y-z- 22
		{ 12, 2, 3, 6, 7,10,11,12,13,14,15,16,18}, // x+y-z+ 23
		{ 12, 2, 4, 5, 8, 9,11,12,13,14,15,17,18}, // x+y+z- 24
		{ 12, 2, 4, 6, 8,10,11,12,13,14,16,17,18}  // x+y+z+ 25
	};
	
	// Specifies whether node has a face on x/y/z boundary, and sign of its inward normal
	int inwardNormals[26][3] = {
		{ 1, 0, 0}, // has face where x-axis points inwards
		{-1, 0, 0}, // x-axis points outwards
		{ 0, 1, 0},
		{ 0,-1, 0},
		{ 0, 0, 1},
		{ 0, 0,-1},
		
		{ 1, 1, 0}, // edge where x and y axis point inwards
		{ 1,-1, 0},
		{ 1, 0, 1},
		{ 1, 0,-1},
		{-1, 1, 0},
		{-1,-1, 0},
		{-1, 0, 1},
		{-1, 0,-1},
		{ 0, 1, 1},
		{ 0, 1,-1},
		{ 0,-1, 1},
		{ 0,-1,-1},
	 	
		{ 1, 1, 1}, 
		{ 1, 1,-1},
		{ 1,-1, 1},
		{ 1,-1,-1},
		{-1, 1, 1},
		{-1, 1,-1},
		{-1,-1, 1},
		{-1,-1,-1}
	};
	
	int numUnknowns = unknowns[typeBC][0];
	
	// Loop over unknowns
	for (int u=0; u<numUnknowns; u++)
	{
		int i_f = unknowns[typeBC][u+1];
		
		int offset[3];
		// If component of c and inward normals are both non-zero and have same direction, 
		// then we need to get this component from the periodic image in that direction.
		for (int dim=0; dim<3; dim++) {
			int c_a = intDat->BasisVel[i_f][dim];
			int m_a = inwardNormals[typeBC][dim];
			// If both non-zero and same direction
			offset[dim] = ((c_a*m_a*(c_a+m_a))/2)*(N[dim]-2); 
		}
		//printf("i_k,typeBC,i,i_f,offset %d,%d,%d,%d %d,%d,%d\n", 
		//i_k, typeBC, i_1D, i_f, offset[0], offset[1], offset[2]);
		
		int periodic_1D = i_1D + offset[0] + N[0]*(offset[1] + N[1]*offset[2]);
		
		// Read component i_f from periodic-offset cell, and write to this one
		f_s[i_f*N_C + i_1D] = f_s[i_f*N_C + periodic_1D];
	}
}

__kernel void GPU_boundary_velocity(
	__global float* f_s,
	__global int_param_struct* intDat,
	__global flp_param_struct* flpDat,
	int wallAxis) // Could try adding this argument to intDat
{
	// Get 3D indices
	int i_3[3]; // Needs to be an array
	i_3[0] = get_global_id(0); // Using global_work_offset = {1,1,1}
	i_3[1] = get_global_id(1);
	i_3[2] = get_global_id(2);
	int i_lu = get_global_id(wallAxis)-1; //  0 or 1 for lower or upper wall
	int i_lu_pm = i_lu*2 - 1;             // -1 or 1 for lower or upper wall
	
	int N[3];
	N[0] = intDat->LatticeSize[0]; 
	N[1] = intDat->LatticeSize[1];
	N[2] = intDat->LatticeSize[2];
	int N_C = N[0]*N[1]*N[2];

	// Wall index (-x,+x, -y,+y, -z,+z) numbered from 0 to 5
	int i_w = wallAxis*2 + i_lu; // -1 because of work_offset
	
	// Eqivalent index for entire lattice (1->1, 2->n-2)
	i_3[wallAxis] += (i_3[wallAxis]-1)*(N[wallAxis]-4);
	int i_1D = i_3[0] + N[0]*(i_3[1] + N[1]*i_3[2]);
	
	// Tabulated velocity boundary condition
	// 5 unknowns and 14 knowns in a symmetric order (see thesis)
	int tabUn[6][5] = { 
		{1, 7, 8, 9,10}, // x- wall
		{2,11,12,13,14}, // x+ wall
		{3, 7,11,15,16}, // y- wall
		{4, 8,12,17,18}, // y+ wall
		{5, 9,13,15,17}, // z- wall
		{6,10,14,16,18}  // z+ wall
	};
	
	int tabKn[6][14] = { 
		{0, 3, 4, 5, 6,15,16,17,18, 2,11,12,13,14},
		{0, 3, 4, 5, 6,15,16,17,18, 1, 7, 8, 9,10},
		{0, 1, 2, 5, 6, 9,10,13,14, 4, 8,12,17,18},
		{0, 1, 2, 5, 6, 9,10,13,14, 3, 7,11,15,16},
		{0, 1, 2, 3, 4, 7, 8,11,12, 6,10,14,16,18},
		{0, 1, 2, 3, 4, 7, 8,11,12, 5, 9,13,15,17}
	};
	
	int tabAxes[6][3] = {
		//n a1 a2
		{0, 1, 2},
		{0, 1, 2},
		{1, 0, 2},
		{1, 0, 2},
		{2, 0, 1},
		{2, 0, 1},
	};

	// Read in 14 knowns
	float f_k[14];
	for (int i_k=0; i_k<14; i_k++) {
		f_k[i_k] = f_s[ i_1D + tabKn[i_w][i_k]*N_C ];
	}	
	
	// Read in velocities
	float u_w[6];
	u_w[0] = flpDat->VelLower[0]; // Only really need 3 of these
	u_w[1] = flpDat->VelLower[1];
	u_w[2] = flpDat->VelLower[2];
	u_w[3] = flpDat->VelUpper[0];
	u_w[4] = flpDat->VelUpper[1];
	u_w[5] = flpDat->VelUpper[2];
	
	float u[3]; // Velocity for this node
	u[0] = u_w[i_lu*3    ];
	u[1] = u_w[i_lu*3 + 1];
	u[2] = u_w[i_lu*3 + 2];
	
	float u_n = -i_lu_pm*u[tabAxes[i_w][0]]; // Inwards normal fluid velocity
	float u_a1 = u[tabAxes[i_w][1]]; // Tangential velocity axis 1 
	float u_a2 = u[tabAxes[i_w][2]]; // Tangential velocity axis 2
	
	// Calculate rho
	float rho = ( f_k[0]+f_k[1]+f_k[2]+f_k[3]+f_k[4]+f_k[5]+f_k[6]+f_k[7]+f_k[8]
		+ 2*(f_k[9]+f_k[10]+f_k[11]+f_k[12]+f_k[13]) )/(1.0f-u_n);
	
	// Calculate 'tranverse momentum correction'
	float N_a1 = 0.5f*(f_k[1]+f_k[5]+f_k[6] -f_k[2]-f_k[7]-f_k[8]) - rho*u_a1/3.0f;
	float N_a2 = 0.5f*(f_k[3]+f_k[5]+f_k[7] -f_k[4]-f_k[6]-f_k[8]) - rho*u_a2/3.0f;
	
	// Calculate unknown normal to wall, write to f_s for this node
	f_s[ i_1D + tabUn[i_w][0]*N_C ] = f_k[9] + rho*u_n/3.0f;
	
	// Other four unknowns
	f_s[ i_1D + tabUn[i_w][1]*N_C ] = f_k[11] + rho*(u_n + u_a1)/6.0f - N_a1;
	f_s[ i_1D + tabUn[i_w][2]*N_C ] = f_k[10] + rho*(u_n - u_a1)/6.0f + N_a1;
	f_s[ i_1D + tabUn[i_w][3]*N_C ] = f_k[13] + rho*(u_n + u_a2)/6.0f - N_a2;
	f_s[ i_1D + tabUn[i_w][4]*N_C ] = f_k[12] + rho*(u_n - u_a2)/6.0f + N_a2;
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

void stream_locations(int N_x, int N_y, int N_z, int i_x, int i_y, int i_z, int* index)
{
	int i_1D = i_x + N_x*(i_y + N_y*i_z); 
	int N_xy = N_x*N_y;
	int N_C = N_xy*N_z;
	
	index[0]  =          i_1D;
	index[1]  =    N_C + i_1D + 1;
	index[2]  =  2*N_C + i_1D - 1; 
	index[3]  =  3*N_C + i_1D     + N_x;
	index[4]  =  4*N_C + i_1D     - N_x;
	index[5]  =  5*N_C + i_1D           + N_xy;
	index[6]  =  6*N_C + i_1D           - N_xy;
	index[7]  =  7*N_C + i_1D + 1 + N_x;
	index[8]  =  8*N_C + i_1D + 1 - N_x;
	index[9]  =  9*N_C + i_1D + 1       + N_xy;
	index[10] = 10*N_C + i_1D + 1       - N_xy;
	index[11] = 11*N_C + i_1D - 1 + N_x;
	index[12] = 12*N_C + i_1D - 1 - N_x;
	index[13] = 13*N_C + i_1D - 1       + N_xy;
	index[14] = 14*N_C + i_1D - 1       - N_xy;
	index[15] = 15*N_C + i_1D     + N_x + N_xy;
	index[16] = 16*N_C + i_1D     + N_x - N_xy;
	index[17] = 17*N_C + i_1D     - N_x + N_xy;
	index[18] = 18*N_C + i_1D     - N_x - N_xy;
}

void equilibirum_distribution_D3Q19(float* f_eq, float rho, float u_x, float u_y, float u_z)
{
	float u_sq = u_x*u_x + u_y*u_y + u_z*u_z;
	
	f_eq[0] = (rho/3.0f)*(1.0f - 1.5f*u_sq);
	
	// could remove factor of 1.5
	f_eq[1]  = (rho/18.0f)*(1.0f + 3.0f*u_x + 4.5f*u_x*u_x - 1.5f*u_sq);
	f_eq[2]  = (rho/18.0f)*(1.0f - 3.0f*u_x + 4.5f*u_x*u_x - 1.5f*u_sq);
	f_eq[3]  = (rho/18.0f)*(1.0f + 3.0f*u_y + 4.5f*u_y*u_y - 1.5f*u_sq);
	f_eq[4]  = (rho/18.0f)*(1.0f - 3.0f*u_y + 4.5f*u_y*u_y - 1.5f*u_sq);
	f_eq[5]  = (rho/18.0f)*(1.0f + 3.0f*u_z + 4.5f*u_z*u_z - 1.5f*u_sq);
	f_eq[6]  = (rho/18.0f)*(1.0f - 3.0f*u_z + 4.5f*u_z*u_z - 1.5f*u_sq);
	
	f_eq[7]  = (rho/36.0f)*(1.0f + 3.0f*(u_x+u_y) + 4.5f*(u_x+u_y)*(u_x+u_y) - 1.5f*u_sq);
	f_eq[8]  = (rho/36.0f)*(1.0f + 3.0f*(u_x-u_y) + 4.5f*(u_x-u_y)*(u_x-u_y) - 1.5f*u_sq);
	f_eq[9]  = (rho/36.0f)*(1.0f + 3.0f*(u_x+u_z) + 4.5f*(u_x+u_z)*(u_x+u_z) - 1.5f*u_sq);
	f_eq[10] = (rho/36.0f)*(1.0f + 3.0f*(u_x-u_z) + 4.5f*(u_x-u_z)*(u_x-u_z) - 1.5f*u_sq);
	
	f_eq[11] = (rho/36.0f)*(1.0f + 3.0f*(-u_x+u_y) + 4.5f*(-u_x+u_y)*(-u_x+u_y) - 1.5f*u_sq);
	f_eq[12] = (rho/36.0f)*(1.0f + 3.0f*(-u_x-u_y) + 4.5f*(-u_x-u_y)*(-u_x-u_y) - 1.5f*u_sq);
	f_eq[13] = (rho/36.0f)*(1.0f + 3.0f*(-u_x+u_z) + 4.5f*(-u_x+u_z)*(-u_x+u_z) - 1.5f*u_sq);
	f_eq[14] = (rho/36.0f)*(1.0f + 3.0f*(-u_x-u_z) + 4.5f*(-u_x-u_z)*(-u_x-u_z) - 1.5f*u_sq);
	
	f_eq[15] = (rho/36.0f)*(1.0f + 3.0f*(u_y+u_z) + 4.5f*(u_y+u_z)*(u_y+u_z) - 1.5f*u_sq);
	f_eq[16] = (rho/36.0f)*(1.0f + 3.0f*(u_y-u_z) + 4.5f*(u_y-u_z)*(u_y-u_z) - 1.5f*u_sq);
	f_eq[17] = (rho/36.0f)*(1.0f + 3.0f*(-u_y+u_z) + 4.5f*(-u_y+u_z)*(-u_y+u_z) - 1.5f*u_sq);
	f_eq[18] = (rho/36.0f)*(1.0f + 3.0f*(-u_y-u_z) + 4.5f*(-u_y-u_z)*(-u_y-u_z) - 1.5f*u_sq);
	
}

void guo_body_force_term(float u_x, float u_y, float u_z,
	float g_x, float g_y, float g_z, float* fGuo)
{
	float uDg = u_x*g_x + u_y*g_y + u_z*g_z;
	
	fGuo[0 ] = -uDg;
	fGuo[1 ] = ( g_x - uDg + 3.0f*u_x*g_x )/6.0f; // Factor of 3 cancelled
	fGuo[2 ] = (-g_x - uDg + 3.0f*u_x*g_x )/6.0f;
	fGuo[3 ] = ( g_y - uDg + 3.0f*u_y*g_y )/6.0f;
	fGuo[4 ] = (-g_y - uDg + 3.0f*u_y*g_y )/6.0f;
	fGuo[5 ] = ( g_z - uDg + 3.0f*u_z*g_z )/6.0f;
	fGuo[6 ] = (-g_z - uDg + 3.0f*u_z*g_z )/6.0f;

	fGuo[7 ] = ( g_x+g_y - uDg + 3.0f*( u_x+u_y)*( g_x+g_y) )/12.0f;
	fGuo[8 ] = ( g_x-g_y - uDg + 3.0f*( u_x-u_y)*( g_x-g_y) )/12.0f;
	fGuo[9 ] = ( g_x+g_z - uDg + 3.0f*( u_x+u_z)*( g_x+g_z) )/12.0f;
	fGuo[10] = ( g_x-g_z - uDg + 3.0f*( u_x-u_z)*( g_x-g_z) )/12.0f;
	
	fGuo[11] = (-g_x+g_y - uDg + 3.0f*(-u_x+u_y)*(-g_x+g_y) )/12.0f;
	fGuo[12] = (-g_x-g_y - uDg + 3.0f*(-u_x-u_y)*(-g_x-g_y) )/12.0f; // could simplify further
	fGuo[13] = (-g_x+g_z - uDg + 3.0f*(-u_x+u_z)*(-g_x+g_z) )/12.0f;
	fGuo[14] = (-g_x-g_z - uDg + 3.0f*(-u_x-u_z)*(-g_x-g_z) )/12.0f;
	
	fGuo[15] = ( g_y+g_z - uDg + 3.0f*( u_y+u_z)*( g_y+g_z) )/12.0f;
	fGuo[16] = ( g_y-g_z - uDg + 3.0f*( u_y-u_z)*( g_y-g_z) )/12.0f;
	fGuo[17] = (-g_y+g_z - uDg + 3.0f*(-u_y+u_z)*(-g_y+g_z) )/12.0f;
	fGuo[18] = (-g_y-g_z - uDg + 3.0f*(-u_y-u_z)*(-g_y-g_z) )/12.0f;
	
}

