#include "struct_header_CL.h"

void equilibirum_distribution_D3Q19(float* f_eq, float rho, float u_x, float u_y, float u_z);
void stream_locations(int n_x, int n_y, int n_z, int i_x, int i_y, int i_z, int* ind);
void guo_body_force_term(float u_x, float u_y, float u_z,
	float g_x, float g_y, float g_z, float* fGuo, float tau);

__kernel void GPU_newtonian_collide_stream_D3Q19_SRT(
	__global int* f_c, 
	__global int* f_s,
	__global float* g, 
	__global float* u, 
	__constant int_param_struct* intDat,
	__constant float_param_struct* flpDat,
	float newtonianTau)
{
	// Get 3D indices
	int i_x = get_global_id(0) + 1; // Shifted by 1 because of propagation buffer layer 
	int i_y = get_global_id(1) + 1;
	int i_z = get_global_id(2) + 1;
	
	int n_x = get_global_size(0);
	int n_y = get_global_size(1);
	int n_z = get_global_size(2);
	
	// 1D index
	int i_1D = i_x + n_x*(i_y + n_y*i_z);
	int n_C = (n_x+2)*(n_y+2)*(n_z+2); // Total elements in f_c (could use sizeof?)
	
	// Read in f (from f_c) for this cell
	float f[19]; 
	
	// Read f_c from __global to private memory (should be coalesced memory access)
	float rho = 0.0;
	for (int i=0; i<19; i++) {
		f[i] = f_c[ i_1D + i*n_C ];
		rho += f[i];
	}
	
	float g_x = g[ i_1D         ];
	float g_y = g[ i_1D +   n_C ];
	float g_z = g[ i_1D + 2*n_C ];
	
	// Compute velocity	(J. Stat. Mech. (2010) P01018 convention)
	float u_x = (f[1]-f[2]+f[7]+f[8] +f[9] +f[10]-f[11]-f[12]-f[13]-f[14] + 0.5*g_x)/rho;
	float u_y = (f[3]-f[4]+f[7]-f[8] +f[11]-f[12]+f[15]+f[16]-f[17]-f[18] + 0.5*g_y)/rho;
	float u_z = (f[5]-f[6]+f[9]-f[10]+f[13]-f[14]+f[15]-f[16]+f[17]-f[18] + 0.5*g_z)/rho;
	
	// Write to __global *u
	u[ i_1D         ] = u_x;
	u[ i_1D +   n_C ] = u_y;
	u[ i_1D + 2*n_C ] = u_z;
	
	// Single relaxtion time (BGK) collision
	float f_eq[19];
	equilibirum_distribution_D3Q19(f_eq, rho, u_x, u_y, u_z);
	
	int streamIndex[19];
	stream_locations(n_x, n_y, n_z, i_x, i_y, i_z, streamIndex);
		
	// Guo, Zheng & Shi body force term (2002)
	float fGuo[19];
	guo_body_force_term(u_x, u_y, u_z, g_x, g_y, g_z, fGuo, newtonianTau);
	
	
	// Propagate to f_s
	float tau = newtonianTau;
	
	for (int i=0; i<19; i++) {
		f_s[streamIndex[i]] = f[i] + (f_eq[i]-f[i])/tau;
	}
	
}

__kernel void GPU_boundary_redirect(__global int* f_s,
	__constant int* streamMapping,
	__constant int_param_struct* intDat,
	__constant float_param_struct* flpDat)
{
	
}

__kernel void GPU_boundary_velocity(
	__global int* f_s,
	__constant int_param_struct* intDat,
	__constant float_param_struct* flpDat,
	int wallAxis)
{
	// Get 3D indices
	int i_3[3];
	i_3[0] = get_global_id(0) + 1; // Shifted by 1 because of propagation buffer layer 
	i_3[1] = get_global_id(1) + 1;
	i_3[2] = get_global_id(2) + 1;

	// Wall index (-x,+x, -y,+y, -z,+z)
	int i_w = wallAxis*2 + i_3[wallAxis]; // wallAxis = (0||1||2) for (x||y||z)		
	
	// Eqivalent index for entire lattice (0->0, 1->n)
	i_3[wallAxis] *= intDat->LatticeSize[wallAxis]; 
	
	int i_1D = i_3[0] + intDat->LatticeSize[0]*(i_3[1] + intDat->LatticeSize[1]*i_3[2]);
	int n_C = intDat->LatticeSize[0]*intDat->LatticeSize[1]*intDat->LatticeSize[2]; 
	
	// Read in local value of f
	// Technically only 14 out of 19 are needed, other 5 will be overwritten 
	float f[19];
	for (int i=0; i<19; i++) {
		f[i] = f_s[ i_1D + i*n_C ];
	}
	
	// Mapping arrays for velocity boundary equations
	// Could test against if statements or in a __constant buffer
	
	// rho = sum(f[1:9]) + 2*sum(f[10:14]) of these values of f 
	int rhoEq[6][14] = {
		{0,  3,  4,  5,  6, 15, 16, 17, 18,  2, 11, 12, 13, 14}, // -x wall
		{0,  3,  4,  5,  6, 15, 16, 17, 18,  1,  7,  8,  9, 10}, // +x
		{0,  1,  2,  5,  6,  9, 10, 13, 14,  4,  8, 12, 17, 18}, // -y
		{0,  1,  2,  5,  6,  9, 10, 13, 14,  3,  7, 11, 15, 16}, // +y
		{0,  1,  2,  3,  4,  7,  8, 11, 12,  6, 10, 14, 16, 18}, // -z
		{0,  1,  2,  3,  4,  7,  8, 11, 12,  5,  9, 13, 15, 17}  // +z
	};
	
	// N corrections
	//0.5*(f + f + f - f - f - f) - v/3
	int N[6][7] = {
		{  3, 15, 16,  4, 17, 18,   1}, // Nx_y
		{  5, 11, 15,  6, 16, 18,   2}, // Nx_z
		{  1,  9, 10,  2, 13, 14,   0}, // Ny_x
		{  5,  9, 13,  6, 10, 14,   2}, // Ny_z
		{  1,  7,  8,  2, 11, 12,   0}, // Nz_x
		{  3,  7, 11,  4,  8, 12,   1}  // Nz_y
	};
		
	// Unknown normal to wall	
	// fi = fj + s*v1*(1/3)*rho      (s = sign)
	int fEq1[6][4] = {
		{1,  2,  1, 0},
		{2,  1, -1, 0},
		{3,  4,  1, 1},
		{4,  3, -1, 1},
		{5,  6,  1, 2},
		{6,  5, -1, 2}
	};
	
	// 4 other unknowns 
	// fi = fj + (v1 + v2)*rho/6 + s*N
	int fEq2[24][7] = {
		{ 7, 12,  1, 0,  1, 1,  -1}, // -x
		{ 8, 11,  1, 0, -1, 1,   1},
		{ 9, 14,  1, 0,  1, 2,  -1},
		{10, 13,  1, 0, -1, 2,   1},
		{11,  8, -1, 0,  1, 1,  -1}, // +x
		{12,  7, -1, 0, -1, 1,   1},
		{13, 10, -1, 0,  1, 2,  -1},
		{14,  9, -1, 0, -1, 2,   1},
		{ 7, 12,  1, 1,  1, 0,  -1}, // -y
		{11,  8,  1, 1, -1, 0,   1},
		{15, 18,  1, 1,  1, 2,  -1},
		{16, 17,  1, 1, -1, 2,   1},
		{ 8, 11, -1, 1,  1, 0,  -1}, // +y
		{12,  7, -1, 1, -1, 0,   1},
		{17, 16, -1, 1,  1, 2,  -1},
		{18, 15, -1, 1, -1, 2,   1},
		{ 9, 14,  1, 2,  1, 0,  -1}, // -z
		{13, 10,  1, 2, -1, 0,   1},
		{15, 18,  1, 2,  1, 1,  -1},
		{17, 16,  1, 2, -1, 1,   1},
		{10, 13, -1, 2,  1, 0,  -1}, // +z
		{14,  9, -1, 2, -1, 0,   1},
		{16, 17, -1, 2,  1, 1,  -1},
		{18, 15, -1, 2, -1, 1,   1}
	};
	
	// Calculate rho
	float rho = 0.0;	
	for (int i=0; i<9; i++) {
		rho += f[rhoEq[i_w][i]];
	}
	for (int i=9; i<14; i++) {
		rho += 2*f[rhoEq[i_w][i]];
	}
		
	// Calculate N corrections 
	float v[6];
	v[0] = flpDat->VelLower[0]; // Only really need 3 of these
	v[1] = flpDat->VelLower[1];
	v[2] = flpDat->VelLower[2];
	v[3] = flpDat->VelUpper[0];
	v[4] = flpDat->VelUpper[1];
	v[5] = flpDat->VelUpper[2];
	
	float v_w[3]; // Velocity for this wall
	v_w[0] = v[i_3[wallAxis]*3];
	v_w[1] = v[i_3[wallAxis]*3] + 1;
	v_w[2] = v[i_3[wallAxis]*3] + 2;
	
	int i_n1 = wallAxis*2;
	int i_n2 = wallAxis*2 + 1;
	
	float N1 = 0.5*(N[i_n1][0] + N[i_n1][1] + N[i_n1][2] - N[i_n1][3] - N[i_n1][4] - N[i_n1][5])
			- rho*v[N[i_n1][6]]/3.0;
	float N2 = 0.5*(N[i_n2][0] + N[i_n2][1] + N[i_n2][2] - N[i_n2][3] - N[i_n2][4] - N[i_n2][5])
			- rho*v[N[i_n2][6]]/3.0;

	// Calculate 5 unknowns and write to f_s*
	// Normal to wall
	f_s[ i_1D + fEq1[i_w][0]*n_C ] = fEq1[i_w][1] + fEq1[i_w][2]*v_w[fEq1[i_w][3]]*rho/3.0;
	
	// Other four
	int r = i_w*4;
	
	f_s[ i_1D + fEq2[r  ][0]*n_C ] = 
		fEq2[r][1] + ( fEq2[r][2]*v_w[fEq2[r][3]] + fEq2[r][4]*v_w[fEq2[r][5]] )*rho/6.0
			+ fEq2[r][2]*N1;
	
	f_s[ i_1D + fEq2[r+1][0]*n_C ] = 
		fEq2[r+1][1] + ( fEq2[r+1][2]*v_w[fEq2[r+1][3]] + fEq2[r+1][4]*v_w[fEq2[r+1][5]] )*rho/6.0
			+ fEq2[r+1][2]*N1;
	
	f_s[ i_1D + fEq2[r+2][0]*n_C ] = 
		fEq2[r+2][1] + ( fEq2[r+2][2]*v_w[fEq2[r+2][3]] + fEq2[r+2][4]*v_w[fEq2[r+2][5]] )*rho/6.0
			+ fEq2[r+2][2]*N2;
	
	f_s[ i_1D + fEq2[r+3][0]*n_C ] = 
		fEq2[r+3][1] + ( fEq2[r+3][2]*v_w[fEq2[r+3][3]] + fEq2[r+3][4]*v_w[fEq2[r+3][5]] )*rho/6.0
			+ fEq2[r+3][2]*N2;
	
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

void stream_locations(int n_x, int n_y, int n_z, int i_x, int i_y, int i_z, int* index)
{
	int i_1D = i_x + n_x*(i_y + n_y*i_z); 
	int n_C = (n_x+2)*(n_y+2)*(n_z+2);
	
	index[0]  =          i_1D;
	index[1]  =    n_C + i_1D + 1;
	index[2]  =  2*n_C + i_1D - 1; 
	index[3]  =  3*n_C + i_1D     + n_x;
	index[4]  =  4*n_C + i_1D     - n_x;
	index[5]  =  5*n_C + i_1D           + n_x*n_y;
	index[6]  =  6*n_C + i_1D           - n_x*n_y;
	index[7]  =  7*n_C + i_1D + 1 + n_x;
	index[8]  =  8*n_C + i_1D + 1 - n_x;
	index[9]  =  9*n_C + i_1D + 1       + n_x*n_y;
	index[10] = 10*n_C + i_1D + 1       - n_x*n_y;
	index[11] = 11*n_C + i_1D - 1 + n_x;
	index[12] = 12*n_C + i_1D - 1 - n_x;
	index[13] = 13*n_C + i_1D - 1       + n_x*n_y;
	index[14] = 14*n_C + i_1D - 1       - n_x*n_y;
	index[15] = 15*n_C + i_1D     + n_x + n_x*n_y;
	index[16] = 16*n_C + i_1D     + n_x - n_x*n_y;
	index[17] = 17*n_C + i_1D     - n_x + n_x*n_y;
	index[18] = 18*n_C + i_1D     - n_x - n_x*n_y;
	
	// Could be adapted to include periodicity
}

void equilibirum_distribution_D3Q19(float* f_eq, float rho, float u_x, float u_y, float u_z)
{
	float u_sq = u_x*u_x + u_y*u_y + u_z*u_z;
	
	f_eq[0] = (rho/3.0)*(1.0 - 1.5*u_sq);
	
	f_eq[1] = (rho/18.0)*(1.0 + 3.0*u_x + 4.5*u_x*u_x - 1.5*u_sq);
	f_eq[2] = (rho/18.0)*(1.0 - 3.0*u_x + 4.5*u_x*u_x - 1.5*u_sq);
	f_eq[3] = (rho/18.0)*(1.0 + 3.0*u_y + 4.5*u_y*u_y - 1.5*u_sq);
	f_eq[4] = (rho/18.0)*(1.0 - 3.0*u_y + 4.5*u_y*u_y - 1.5*u_sq);
	f_eq[5] = (rho/18.0)*(1.0 + 3.0*u_z + 4.5*u_z*u_z - 1.5*u_sq);
	f_eq[6] = (rho/18.0)*(1.0 - 3.0*u_z + 4.5*u_z*u_z - 1.5*u_sq);
	
	f_eq[7] = (rho/36.0)*(1.0 + 3.0*(u_x+u_y) + 4.5*(u_x+u_y)*(u_x+u_y) - 1.5*u_sq);
	f_eq[8] = (rho/36.0)*(1.0 + 3.0*(u_x-u_y) + 4.5*(u_x-u_y)*(u_x-u_y) - 1.5*u_sq);
	f_eq[9] = (rho/36.0)*(1.0 + 3.0*(u_x+u_z) + 4.5*(u_x+u_z)*(u_x+u_z) - 1.5*u_sq);
	f_eq[10] = (rho/36.0)*(1.0 + 3.0*(u_x-u_z) + 4.5*(u_x-u_z)*(u_x-u_z) - 1.5*u_sq);
	
	f_eq[11] = (rho/36.0)*(1.0 + 3.0*(-u_x+u_y) + 4.5*(-u_x+u_y)*(-u_x+u_y) - 1.5*u_sq);
	f_eq[12] = (rho/36.0)*(1.0 + 3.0*(-u_x-u_y) + 4.5*(-u_x-u_y)*(-u_x-u_y) - 1.5*u_sq);
	f_eq[13] = (rho/36.0)*(1.0 + 3.0*(-u_x+u_z) + 4.5*(-u_x+u_z)*(-u_x+u_z) - 1.5*u_sq);
	f_eq[14] = (rho/36.0)*(1.0 + 3.0*(-u_x-u_z) + 4.5*(-u_x-u_z)*(-u_x-u_z) - 1.5*u_sq);
	
	f_eq[15] = (rho/36.0)*(1.0 + 3.0*(u_y+u_z) + 4.5*(u_y+u_z)*(u_y+u_z) - 1.5*u_sq);
	f_eq[16] = (rho/36.0)*(1.0 + 3.0*(u_y-u_z) + 4.5*(u_y-u_z)*(u_y-u_z) - 1.5*u_sq);
	f_eq[17] = (rho/36.0)*(1.0 + 3.0*(-u_y+u_z) + 4.5*(-u_y+u_z)*(-u_y+u_z) - 1.5*u_sq);
	f_eq[18] = (rho/36.0)*(1.0 + 3.0*(-u_y-u_z) + 4.5*(-u_y-u_z)*(-u_y-u_z) - 1.5*u_sq);
	
}

void guo_body_force_term(float u_x, float u_y, float u_z,
	float g_x, float g_y, float g_z, float* fGuo, float tau)
{
	float pf = (1 - 0.5/tau); // Prefactor 
	float ug = u_x*g_x + u_y*g_y + u_z*g_z;
	
	fGuo[0 ] = -pf*ug;
	fGuo[1 ] = (pf/18.0)*(3*( g_x-ug) + 9*u_x*g_x);
	fGuo[2 ] = (pf/18.0)*(3*(-g_x-ug) + 9*u_x*g_x);
	fGuo[3 ] = (pf/18.0)*(3*( g_y-ug) + 9*u_y*g_y);
	fGuo[4 ] = (pf/18.0)*(3*(-g_y-ug) + 9*u_y*g_y);
	fGuo[5 ] = (pf/18.0)*(3*( g_z-ug) + 9*u_z*g_z);
	fGuo[6 ] = (pf/18.0)*(3*(-g_z-ug) + 9*u_z*g_z);

	fGuo[7 ] = (pf/18.0)*(3*( g_x+g_y-ug) + 9*( u_x+u_y)*( g_x+g_y));
	fGuo[8 ] = (pf/18.0)*(3*( g_x-g_y-ug) + 9*( u_x-u_y)*( g_x-g_y));
	fGuo[9 ] = (pf/18.0)*(3*( g_x+g_z-ug) + 9*( u_x+u_z)*( g_x+g_z));
	fGuo[10] = (pf/18.0)*(3*( g_x-g_z-ug) + 9*( u_x-u_z)*( g_x-g_z));
	
	fGuo[11] = (pf/18.0)*(3*(-g_x+g_y-ug) + 9*(-u_x+u_y)*(-g_x+g_y));
	fGuo[12] = (pf/18.0)*(3*(-g_x-g_y-ug) + 9*(-u_x-u_y)*(-g_x-g_y)); // See if removing - is faster
	fGuo[13] = (pf/18.0)*(3*(-g_x+g_z-ug) + 9*(-u_x+u_z)*(-g_x+g_z));
	fGuo[14] = (pf/18.0)*(3*(-g_x-g_z-ug) + 9*(-u_x-u_z)*(-g_x-g_z));
	
	fGuo[15] = (pf/18.0)*(3*( g_y+g_z-ug) + 9*( u_y+u_z)*( g_y+g_z));
	fGuo[16] = (pf/18.0)*(3*( g_y-g_z-ug) + 9*( u_y-u_z)*( g_y-g_z));
	fGuo[17] = (pf/18.0)*(3*(-g_y+g_z-ug) + 9*(-u_y+u_z)*(-g_y+g_z));
	fGuo[18] = (pf/18.0)*(3*(-g_y-g_z-ug) + 9*(-u_y-u_z)*(-g_y-g_z));
	
}



