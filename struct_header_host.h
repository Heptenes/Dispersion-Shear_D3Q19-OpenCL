typedef struct {

	cl_int BasisVel[19][3];	
	cl_int MaxIterations;
	cl_int LatticeSize[3];
	cl_int BoundaryConds[3];
    cl_int ViscosityModel;
	
} int_param_struct;


typedef struct {
	
    cl_float ConstBodyForce[3];
	cl_float EqWeights[19]; 
	cl_float VelUpper[3];
	cl_float VelLower[3];
	cl_float NewtonianTau;
  
} flp_param_struct;
