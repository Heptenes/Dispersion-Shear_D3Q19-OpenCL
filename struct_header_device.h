typedef struct {

	int BasisVel[19][3];	
	int MaxIterations;
	int LatticeSize[3];
	int BoundaryConds[3];
    int ViscosityModel;
	
} int_param_struct;


typedef struct {
	
    float ConstBodyForce[3];
	float EqWeights[19]; 
	float VelUpper[3];
	float VelLower[3];
	float NewtonianTau;
  
} flp_param_struct;
