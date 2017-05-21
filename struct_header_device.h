#define WORD_STRING_SIZE 64
#define USE_CONSTANT_BODY_FORCE
#define DEBUG

typedef struct {

	int BasisVel[19][3];	
	int MaxIterations;
	int LatticeSize[3];
	int BoundaryConds[3];
    int ViscosityModel;
	
} int_param_struct __attribute__((aligned (16)));


typedef struct {
	
    float ConstBodyForce[3];
	float EqWeights[19]; 
	float VelUpper[3];
	float VelLower[3];
	float NewtonianTau;
  
} flp_param_struct __attribute__((aligned (16)));
