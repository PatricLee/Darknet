#include "image.h"
#include <stdlib.h>
#include <pthread.h>

#define OVERSEERMAXPOINTS 100

typedef struct Oversee{
	int level;//the higher, the more you inspect into the learning process
	
	int points;
	int *iteration;
	float *iou_train;
	float *rp_train;
	float *recall_train;
	float *iou_val;
	float *rp_val;
	float *recall_val;
	
	char path[128];//where to store overseer
	
	pthread_t thread;
	pthread_mutex_t pmutex;
	pthread_cond_t pcondition;
	
	image pic;
	
};

struct Oversee overseer;
void *OverseerInitialize(void *parameter);
void OverseerPushPoints(int iteration, float iou_train, float rp_train,
	float recall_train, float iou_val, float rp_val, float recall_val); 
void *OverseerThread(void *parameter);
