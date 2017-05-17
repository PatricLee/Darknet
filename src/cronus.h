#define VAL_TRAIN 1
#define VAL_VAL 2
#define VAL_TRAINVAL 3

#define NMS_THRESH 0.05
#define RP_THRESH 0.001

void cronus_train(char *datacfg, char *weightfile, int *gpus, int ngpus, int clear);
void cronus2_trainmain(char *datacfg, char *weightfile, int *gpus, int ngpus, int clear);
void cronus_recall5_online(network net, char* datacfg, int validate_type, float *result);
void cronus_recall5(char* datacfg, char* weights, int validate_type);
void cronus_test5(char *datacfg, char *weightfile, float thresh, int validate_type);
void cronus_test5_single(char *datacfg, char *weightfile, float thresh, char* filename);