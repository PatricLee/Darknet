#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "overseer.h"

//parameter: int level, char *path
void *OverseerInitialize(void *parameter){
	overseer.level = (int *)parameter;
	if(overseer.level == 0)return -1;
	
	strcpy(overseer.path, ((char *)parameter + sizeof(int)));
	
	overseer.points = 0;
	overseer.iteration = calloc(OVERSEERMAXPOINTS, sizeof(int));
	overseer.iou_train = calloc(OVERSEERMAXPOINTS, sizeof(float));
	overseer.rp_train = calloc(OVERSEERMAXPOINTS, sizeof(float));
	overseer.recall_train = calloc(OVERSEERMAXPOINTS, sizeof(int));
	overseer.iou_val = calloc(OVERSEERMAXPOINTS, sizeof(float));
	overseer.rp_val = calloc(OVERSEERMAXPOINTS, sizeof(float));
	overseer.recall_val = calloc(OVERSEERMAXPOINTS, sizeof(float));
	
	pthread_mutex_init(overseer.pmutex, NULL);
	pthread_cond_init(overseer.pcondition, NULL);
	pthread_create(&(overseer.thread), 0, OverseerThread, NULL);
	
	//wait for thread startup
	Sleep(1000);
	
	pthread_mutex_lock(&(overseer.pmutex));
	pthread_cond_signal(&(overseer.pcondition));
	pthread_mutex_unlock(&(overseer.pmutex));
	
	return 0;
}

void OverseerPushPoints(int iteration, float iou_train, float rp_train,
	float recall_train, float iou_val, float rp_val, float recall_val){
	int i;
	
	//overwrite points with older iterations
	for(i=0;i<overseer.points;i++){
		if(overseer.iteration[i]>iteration){
			overseer.points=i;
			break;
		}
	}
	overseer.iteration[overseer.points]=iteration;
	overseer.iou_train[overseer.points]=iou_train;
	overseer.rp_train[overseer.points]=rp_train;
	overseer.recall_train[overseer.points]=recall_train;
	overseer.iou_val[overseer.points]=iou_val;
	overseer.rp_val[overseer.points]=rp_val;
	overseer.recall_val[overseer.points]=recall_val;
	overseer.points++;
	
	//update overseer
	pthread_mutex_lock(&(overseer.pmutex));
	pthread_cond_signal(&(overseer.pcondition));
	pthread_mutex_unlock(&(overseer.pmutex));
	
	return;
}

float FindLargest(float *d, int n){
	float max = 0;
	for(int i=0;i<n;i++){
		if (d[i] > max)max = d[i];
	}
	return max;
}



void OverseerDraw(){
	//first set to while
	SetColor(overseer.pic, 1.0);
	
	//draw coordinate axie
	DrawLine(overseer.pic, 100, 700, 700, 700, 0, 0, 0);
	DrawLine(overseer.pic, 100, 700, 100, 100, 0, 0, 0);
	
	if(overseer.points<=1) return;//cannot draw when less than 2 points
	int max_iteration = overseer.iteration[overseer.points];
	float max_iou = (FindLargest(overseer.iou_train,overseer.points)>
		FindLargest(overseer.iou_val,overseer.points))?
		FindLargest(overseer.iou_train,overseer.points):
		FindLargest(overseer.iou_val,overseer.points);
	float max_rp = (FindLargest(overseer.rp_train,overseer.points)>
		FindLargest(overseer.rp_val,overseer.points))?
		FindLargest(overseer.rp_train,overseer.points):
		FindLargest(overseer.rp_val,overseer.points);
	float max_recall = (FindLargest(overseer.recall_train,overseer.points)>
		FindLargest(overseer.recall_val,overseer.points))?
		FindLargest(overseer.recall_train,overseer.points):
		FindLargest(overseer.recall_val,overseer.points);
	int x1,x2,y1,y2;
	for(int i=0;i<(overseer.points - 1);i++){
		x1 = (int)(600*((float)(overseer.iteration[i])/(float)max_iteration))+100;
		x2 = (int)(600*((float)(overseer.iteration[i+1])/(float)max_iteration))+100;
		
		y1 = (int)((float)600*(1 - overseer.iou_train[i]/max_iou))+100;
		y2 = (int)((float)600*(1 - overseer.iou_train[i+1]/max_iou))+100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 0.5, 0, 0);

		y1 = (int)((float)600 * (1 - overseer.iou_val[i] / max_iou)) + 100;
		y2 = (int)((float)600 * (1 - overseer.iou_val[i + 1] / max_iou)) + 100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 1, 0, 0);

		y1 = (int)((float)600 * (1 - overseer.rp_train[i] / max_rp)) + 100;
		y2 = (int)((float)600 * (1 - overseer.rp_train[i + 1] / max_rp)) + 100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 0, 0.5, 0);

		y1 = (int)((float)600 * (1 - overseer.rp_val[i] / max_rp)) + 100;
		y2 = (int)((float)600 * (1 - overseer.rp_val[i + 1] / max_rp)) + 100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 0, 1, 0);

		y1 = (int)((float)600 * (1 - overseer.recall_train[i] / max_recall)) + 100;
		y2 = (int)((float)600 * (1 - overseer.recall_train[i + 1] / max_recall)) + 100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 0, 0, 0.5);

		y1 = (int)((float)600 * (1 - overseer.recall_val[i] / max_recall)) + 100;
		y2 = (int)((float)600 * (1 - overseer.recall_val[i + 1] / max_recall)) + 100;
		DrawLine(overseer.pic, x1, y1, x2, y2, 0, 0, 1);
	}
	
	//waitKey(50);
}

void *OverseerThread(void *parameter){
	//lock mutex before ready
	pthread_mutex_lock(&(overseer.pmutex));
	overseer.pic = make_image(800, 800, 3);
	
	//try loading overseer file
	FILE *overseer_file;
	int iteration;
	float iou_t, rp_t, recall_t, iou_v, rp_v, recall_v;
	overseer_file = fopen(overseer.path, "r");
	if(overseer_file){
		overseer.points=0;
		while(fscanf(overseer_file, "%d %f %f %f %f %f %f",&iteration, &iou_t,&rp_t,&recall_t,&iou_v,&rp_v,&recall_v == 7)){
			overseer.iteration[overseer.points]=iteration;
			overseer.iou_train[overseer.points]=iou_t;
			overseer.rp_train[overseer.points]=rp_t;
			overseer.recall_train[overseer.points]=recall_t;
			overseer.iou_val[overseer.points]=iou_v;
			overseer.rp_val[overseer.points]=rp_v;
			overseer.recall_val[overseer.points]=recall_v;
			overseer.points++;
		}
		fclose(overseer_file);
	}
	pthread_mutex_unlock(&overseer.pmutex);
	
	int i;
	while(1){
		pthread_mutex_lock(&(overseer.pmutex));
		pthread_cond_wait(&(overseer.pcondition),&(overseer.pmutex));
		
		overseer_file = fopen(overseer.path, "w");
		if(!overseer_file) file_error(overseer.path);
		for(i=0;i<overseer.points;i++){
			fprintf(overseer_file, "%d %f %f %f %f %f %f\n", overseer.iteration[i],
				overseer.iou_train[i],overseer.rp_train[i],overseer.recall_train[i],
				overseer.iou_val[i],overseer.rp_val[i],overseer.recall_val[i]);
		}
		fclose(overseer_file);
		
#ifdef OPENCV
		OverseerDraw();
#endif
		
		pthread_mutex_unlock(&(overseer.pmutex));
	}
}