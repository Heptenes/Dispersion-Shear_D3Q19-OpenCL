#define WORD_STRING_SIZE 64

typedef struct {
	
    char keyword[WORD_STRING_SIZE];
	int dataType;
	void* varPtr;
	char defString[WORD_STRING_SIZE];
	
} input_data_struct;


typedef struct {

	cl_int BasisVel[19][3];
	
	cl_int MaxIterations;
	
	cl_int LatticeSize[3];
	
	cl_int BoundaryConds[6];
  
    cl_int ViscosityModel;
	
} int_param_struct;


typedef struct {
	
    cl_float ConstBodyForce[3];
	cl_float EqWeights[19]; 
	cl_float VelUpper[3];
	cl_float VelLower[3];
  
} float_param_struct;

// Params which don't need to be passed to GPU
typedef struct {
	
	int consolePrintFreq;
    char initialDist[WORD_STRING_SIZE];
  
} host_param_struct;

