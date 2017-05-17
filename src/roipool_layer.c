#include "roipool_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

extern roi_boxes *roiboxes;

//roi pooling takes as intake hxwxc feature map and outputs no more than n sizexsize poolings
//rois come in as 2D boxes with left,right,top,down
roipool_layer make_roipool_layer(int batch, int h, int w, int c, int n, int size, int padding){
	roipool_layer l={0};
	l.batch = batch;
	l.w = w;
	l.h = h;
	l.c = c;//same for input and output
	l.out_h = size;
	l.out_w = size;
	l.out_c = c;
	l.max_boxes = n;//per batch
	l.size = size;//output feature map size, roipool has no filter size
	//l.pad = padding;
	l.truths = 30 * 5;
	l.inputs = h*w*c;
	l.outputs = n*size*size*c;//per batch
	l.output = calloc(l.outputs*l.batch, sizeof(float));
	l.indexes = calloc(l.outputs*l.batch, sizeof(int));
	l.delta = calloc(l.outputs*l.batch, sizeof(float));
	
	l.roiboxes = roiboxes;//global, share with region layer 7
	if(!l.roiboxes){
		l.roiboxes = calloc(1, sizeof(roi_boxes));
		roiboxes->roibox = calloc(l.max_boxes*l.batch, sizeof(roi_box));
		roiboxes->n = calloc(l.batch, sizeof(int));
		roiboxes->prob = calloc(l.max_boxes*l.batch, sizeof(float));
	}
	
	l.forward = forward_roipool_layer;
	l.backward = backward_roipool_layer;
#ifdef GPU
	//l.forward_gpu = forward_roipool_gpu;
	//l.backward_gpu = backward_roipool_gpu;
	l.indexes_gpu = cuda_make_int_array(l.outputs*l.batch);
	l.output_gpu = cuda_make_array(l.output, l.outputs*l.batch);
	l.delta_gpu = cuda_make_array(l.delta, l.outputs*l.batch);
#endif
	fprintf(stderr, "roi          %d x %d x %d  %4d x%4d x%4d\n", n ,size, size, w, h, c);
	return l;
}

void forward_roipool_layer(const roipool_layer l, network_state state){
	int b, n, i, j, c, pa, pb;
	int index;
	//copy roi boxes from truth, if necessary
	if(!roiboxes){
		for(b=0; b<l.batch; b++){
			for(n=0; n<l.max_boxes; n++){
				l.roiboxes->n[b] = n;
				if(!(*(float *)(state.truth + b*l.truths + n*5))) break;
				memcpy(&(l.roiboxes->roibox[b*l.max_boxes + n]), (float *)(state.truth + b*l.truths + n * 5), 4 * sizeof(float));
			}
		}
	}
	roi_box t;
	for(b = 0; b < l.batch; b++){
		for(n = 0; n < l.roiboxes->n[b]; n++){
			if(n == l.max_boxes){
				l.roiboxes->n[b] = l.max_boxes;
				break;
			}
			t = l.roiboxes->roibox[b*l.max_boxes + n];
			float left, right, top, down;
			left = t.x - t.w/2;
			right = t.x + t.w/2;
			top = t.y - t.w/2;
			down = t.y + t.w/2;
			float pw = t.w/l.size;
			float ph = t.h/l.size;
			//if roi window is too small
			if(pw <= 1.0/l.w || ph <= 1.0/l.h) continue;
			for(i = 0; i < l.size; i++){
				for(j = 0; j < l.size; j++){
					int pcellleft = (int)floor((left + j*pw)*l.w);
					int pcellright = (int)ceil((left + (j+1)*pw)*l.w);
					int pcelltop = (int)floor((top + i*ph)*l.h);
					int pcelldown = (int)ceil((top + (i+1)*ph)*l.h);
					if(pcellright<=pcellleft||pcelldown<=pcelltop){
						for (c = 0; c < l.c; c++){
							int output_index = ((b*l.c + c)*l.out_h + i)*l.out_w + j;
							l.output[output_index] = -FLT_MAX;
							l.indexes[output_index] = -1;
						}
					}
					else{
						for(c = 0; c < l.c; c++){
							//max pooling
							float max_v = -FLT_MAX;
							int max_i = -1;
							int output_index = ((b*l.c + c)*l.out_h + i)*l.out_w + j;
							for(pa = pcellleft; pa<pcellright; pa++){
								for (pb = pcelltop; pb<pcelldown; pb++){
									int index = ((b*l.c + c)*l.out_h + pb)*l.out_w + pa;
									float val = state.input[index];
									if(val>max_v){
										val = max_v;
										max_i = index;
									}
								}
							}
							l.output[output_index] = max_v;
							l.indexes[output_index] = max_i;
						}
					}
				}
			}
		}
	}
	if(!state.train)return;
	//convert state.truth, remove box coords
	
	
	return;
}

void backward_roipool_layer(const roipool_layer l, network_state state){
	int b, n;
	for(b = 0; b < l.batch; b++){
		
	}
	return;
}