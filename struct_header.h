
typedef struct {

	int basisVel[19][3];
    int boundaryConds[6];
	int totalLatticeSize[3];
  
} int_param_struct;


typedef struct {
	
    float constBodyForce;
	float eqWeights[19]; 
  
} float_param_struct;
