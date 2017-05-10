#define WORD_STRING_SIZE 64

typedef struct {
	
    char keyword[WORD_STRING_SIZE];
	int dataType;
	void* varPtr;
	char defString[WORD_STRING_SIZE];
	
} input_data_struct;


typedef struct {

	int BasisVel[19][3];
	
	int MaxIterations;
	
	int LatticeSize[3];
	
	int BoundaryConds[6];
  
    int ViscosityModel;
	
} int_param_struct;


typedef struct {
	
    float ConstBodyForce[3];
	float EqWeights[19]; 
	float VelUpper[3];
	float VelLower[3];
  
} float_param_struct;

// Params which don't need to be passed to GPU
typedef struct {
	
    char initialDist[WORD_STRING_SIZE];
  
} host_param_struct;

