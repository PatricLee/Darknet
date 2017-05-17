#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords)
{
    region_layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
    l.truths = 30*(5);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}
region_layer5 make_region_layer5(int batch, int w, int h, int n, int classes, int coords)
{
	region_layer5 l = { 0 };
	l.type = REGION5;

	l.n = n;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.classes = classes;
	l.coords = coords;
	l.cost = calloc(1, sizeof(float));
	l.biases = calloc(n * 2, sizeof(float));
	l.bias_updates = calloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + coords + 1);
	l.inputs = l.outputs;
	l.truths = 30 * (coords + 1);
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < n * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_region_layer5;
	l.backward = backward_region_layer5;
#ifdef GPU
	l.forward_gpu = forward_region_layer5_gpu;
	l.backward_gpu = backward_region_layer5_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}
region_layer7 make_region_layer7(int batch, int w, int h, int n, int classes, int coords)
{
	region_layer7 l = { 0 };
	l.type = REGION7;

	l.n = n;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.classes = classes;
	l.coords = coords;
	l.cost = calloc(1, sizeof(float));
	l.biases = calloc(n * 2, sizeof(float));
	l.bias_updates = calloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + coords + 1);
	l.inputs = l.outputs;
	l.truths = 30 * (coords + 1);
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < n * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_region_layer7;
	l.backward = backward_region_layer7;
#ifdef GPU
	l.forward_gpu = forward_region_layer7_gpu;
	l.backward_gpu = backward_region_layer7_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "region7\n");
	srand(0);

	return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    if(DOABS){
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}
box2 get_region_box5(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	box2 b;
	b.x = (i + logistic_activate(x[index + 0])) / w;
	b.y = (j + logistic_activate(x[index + 1])) / h;
	b.w = exp(x[index + 2]) * biases[2 * n];
	b.h = exp(x[index + 3]) * biases[2 * n + 1];
	b.rz = PI / 2 * tanh_activate(x[index + 4]);
	if (DOABS) {
		b.w = exp(x[index + 2]) * biases[2 * n] / w;
		b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
	}
	return b;
}
box3 get_region_box7(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
	box3 b;
	b.x = (i + logistic_activate(x[index + 0])) / w;
	b.y = (j + logistic_activate(x[index + 1])) / h;
	b.w = exp(x[index + 2]) * biases[2 * n];
	b.h = exp(x[index + 3]) * biases[2 * n + 1];
	b.rz = PI / 2 * tanh_activate(x[index + 4]);
	b.z = logistic_activate(x[index + 5]);
	b.l = logistic_activate(x[index + 6]);
	if (DOABS) {
		b.w = exp(x[index + 2]) * biases[2 * n] / w;
		b.h = exp(x[index + 3]) * biases[2 * n + 1] / h;
	}
	return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w / biases[2*n]);
    float th = log(truth.h / biases[2*n + 1]);
    if(DOABS){
        tw = log(truth.w*w / biases[2*n]);
        th = log(truth.h*h / biases[2*n + 1]);
    }

    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    return iou;
}
float delta_region_box5(box2 truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
	box2 pred = get_region_box5(x, biases, n, index, i, j, w, h);
	float iou = box_iou5(pred, truth);

	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	float tw = log(truth.w / biases[2 * n]);
	float th = log(truth.h / biases[2 * n + 1]);
	float trz = truth.rz;
	if (DOABS) {
		tw = log(truth.w*w / biases[2 * n]);
		th = log(truth.h*h / biases[2 * n + 1]);
	}

	delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
	delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
	delta[index + 2] = scale * (tw - x[index + 2]);
	delta[index + 3] = scale * (th - x[index + 3]);
	delta[index + 4] = scale * sinf(trz - PI / 2 * tanh_activate(x[index + 4]))*cosf(trz - PI / 2 * tanh_activate(x[index + 4]))*PI*tanh_gradient(tanh_activate(x[index + 4]));
	return iou;
}
float delta_region_box7(box3 truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
	box3 pred = get_region_box7(x, biases, n, index, i, j, w, h);
	float iou = box_iou7(pred, truth);

	float tx = (truth.x*w - i);
	float ty = (truth.y*h - j);
	float tw = log(truth.w / biases[2 * n]);
	float th = log(truth.h / biases[2 * n + 1]);
	float trz = truth.rz;
	float tz = truth.z;
	float tl = truth.l;
	if (DOABS) {
		tw = log(truth.w*w / biases[2 * n]);
		th = log(truth.h*h / biases[2 * n + 1]);
	}

	delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
	delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
	delta[index + 2] = scale * (tw - x[index + 2]);
	delta[index + 3] = scale * (th - x[index + 3]);
	delta[index + 4] = scale * sinf(trz - PI / 2 * tanh_activate(x[index + 4]))*cosf(trz - PI / 2 * tanh_activate(x[index + 4]))*PI*tanh_gradient(tanh_activate(x[index + 4]));
	delta[index + 5] = scale * (tz - logistic_activate(x[index + 5])) * logistic_gradient(logistic_activate(x[index + 5]));
	delta[index + 6] = scale * (tl - logistic_activate(x[index + 6])) * logistic_gradient(logistic_activate(x[index + 6]));
	return iou;
}
//calculate height difference on h demension(iou of z & l in box3)
float box_iou_h7(box3 a, box3 b) {
	float amin = a.z - a.l / 2;
	float amax = a.z + a.l / 2;
	float bmin = b.z - b.l / 2;
	float bmax = b.z + b.l / 2;

	float imax = (amax > bmax) ? bmax : amax;
	float imin = (amin < bmin) ? bmin : amin;
	float omax = (amax > bmax) ? amax : bmax;
	float omin = (amin < bmin) ? amin : bmin;

	if (imax <= imin)return 0;
	return (imax - imin) / (omax - omin);
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, float *avg_cat)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class] = scale * (1 - output[index + class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        for(n = 0; n < classes; ++n){
            delta[index + n] = scale * (((n == class)?1 : 0) - output[index + n]);
            if(n == class) *avg_cat += output[index + n];
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
void forward_region_layer(const region_layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    #endif
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
        }
    }


#ifndef GPU
    if (l.softmax_tree){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    } else if (l.softmax){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    }
#endif
    if(!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(state.truth + t*5 + b*l.truths);
                if(!truth.x) break;
                int class = state.truth[t*5 + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                    float best_iou = 0;
                    int best_class = -1;
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(state.truth + t*5 + b*l.truths);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_class = state.truth[t*5 + b*l.truths + 4];
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if(l.classfix == -1) l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    else{
                        if (best_iou > l.thresh) {
                            l.delta[index + 4] = 0;
                            if(l.classfix > 0){
                                delta_region_class(l.output, l.delta, index + 5, best_class, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + 4] : 1), &avg_cat);
                                ++class_count;
                            }
                        }
                    }

                    if(*(state.net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n];
                        truth.h = l.biases[2*n+1];
                        if(DOABS){
                            truth.w = l.biases[2*n]/l.w;
                            truth.h = l.biases[2*n+1]/l.h;
                        }
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(state.truth + t*5 + b*l.truths);

            if(!truth.x) break;
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n];
                    pred.h = l.biases[2*n+1];
                    if(DOABS){
                        pred.w = l.biases[2*n]/l.w;
                        pred.h = l.biases[2*n+1]/l.h;
                    }
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            }


            int class = state.truth[t*5 + b*l.truths + 4];
            if (l.map) class = l.map[class];
            delta_region_class(l.output, l.delta, best_index + 5, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    #ifndef GPU
    flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    #endif
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}
void forward_region_layer5(const region_layer l, network_state state)
{
	int coords = l.coords;
	int i, j, b, t, n;
	int size = l.coords + l.classes + 1;
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
#ifndef GPU
	flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
#endif
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			l.output[index + coords] = logistic_activate(l.output[index + coords]);
		}
	}


#ifndef GPU
	if (l.softmax_tree) {
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
			}
		}
	}
	else if (l.softmax) {
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
			}
		}
	}
#endif
	if (!state.train) return;
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	float avg_iou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		if (l.softmax_tree) {
			int onlyclass = 0;
			for (t = 0; t < 30; ++t) {
				box2 truth = float_to_box5(state.truth + t * (coords+1) + b*l.truths);
				if (!truth.x) break;
				int class = state.truth[t * (coords+1) + b*l.truths + coords];
				float maxp = 0;
				int maxi = 0;
				if (truth.x > 100000 && truth.y > 100000) {
					for (n = 0; n < l.n*l.w*l.h; ++n) {
						int index = size*n + b*l.outputs + coords + 1;
						float scale = l.output[index - 1];
						float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class);
						if (p > maxp) {
							maxp = p;
							maxi = n;
						}
					}
					int index = size*maxi + b*l.outputs + coords + 1;
					delta_region_class(l.output, l.delta, index, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
					++class_count;
					onlyclass = 1;
					break;
				}
			}
			if (onlyclass) continue;
		}
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
					box2 pred = get_region_box5(l.output, l.biases, n, index, i, j, l.w, l.h);
					float best_iou = 0;
					int best_class = -1;
					for (t = 0; t < 30; ++t) {
						box2 truth = float_to_box5(state.truth + t * (coords + 1) + b*l.truths);
						if (!truth.x) break;
						float iou = box_iou5(pred, truth);
						if (iou > best_iou) {
							best_class = state.truth[t * (coords + 1) + b*l.truths + coords];
							best_iou = iou;
						}
					}
					avg_anyobj += l.output[index + coords];
					//l.delta[index + coords] = l.noobject_scale * ((0 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]));
					if (l.classfix == -1) l.delta[index + coords] = l.noobject_scale * ((best_iou - l.output[index + coords]) * logistic_gradient(l.output[index + coords]));
					else {
						if (best_iou > l.thresh) {
							l.delta[index + coords] = l.object_scale * (1 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]);//0;
							if (l.classfix > 0) {
								delta_region_class(l.output, l.delta, index + coords + 1, best_class, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + coords] : 1), &avg_cat);
								++class_count;
							}
						}
						if (best_iou < 0.05) {
							l.delta[index + coords] = l.noobject_scale * (0 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]);
						}
					}

					//not clear what this does
					if (*(state.net.seen) < 6400) {
						box2 truth = { 0 };
						truth.x = (i + .5) / l.w;
						truth.y = (j + .5) / l.h;
						truth.w = l.biases[2 * n];
						truth.h = l.biases[2 * n + 1];
						truth.rz = 0;
						if (DOABS) {
							truth.w = l.biases[2 * n] / l.w;
							truth.h = l.biases[2 * n + 1] / l.h;
						}
						delta_region_box5(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
					}
				}
			}
		}
		for (t = 0; t < 30; ++t) {
			box2 truth = float_to_box5(state.truth + t * (coords + 1) + b*l.truths);

			if (!truth.x) break;
			float best_iou = 0;
			float rz = 0;
			int best_index = 0;
			int best_n = 0;
			i = (truth.x * l.w);
			j = (truth.y * l.h);
			rz = truth.rz;
			//printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
			box2 truth_shift = truth;
			truth_shift.x = 0;
			truth_shift.y = 0;
			truth_shift.rz = truth.rz;
			//printf("index %d %d\n",i, j);
			for (n = 0; n < l.n; ++n) {
				int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
				box2 pred = get_region_box5(l.output, l.biases, n, index, i, j, l.w, l.h);
				if (l.bias_match) {
					pred.w = l.biases[2 * n];
					pred.h = l.biases[2 * n + 1];
					if (DOABS) {
						pred.w = l.biases[2 * n] / l.w;
						pred.h = l.biases[2 * n + 1] / l.h;
					}
				}
				//printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
				pred.x = 0;
				pred.y = 0;
				float iou = box_iou5(pred, truth_shift);
				if (iou > best_iou) {
					best_index = index;
					best_iou = iou;
					best_n = n;
				}
			}
			//printf("%d %f (%f, %f) %f x %f, %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h, truth.rz);

			float iou = delta_region_box5(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
			if (iou > l.thresh) recall += 1;
			avg_iou += iou;

			avg_obj += l.output[best_index + coords];
			/*l.delta[best_index + coords] = l.object_scale * (1 - l.output[best_index + coords]) * logistic_gradient(l.output[best_index + coords]);
			if (l.rescore) {
				l.delta[best_index + coords] = l.object_scale * (iou - l.output[best_index + coords]) * logistic_gradient(l.output[best_index + coords]);
			}*/


			int class = state.truth[t * (coords + 1) + b*l.truths + coords];
			if (l.map) class = l.map[class];
			delta_region_class(l.output, l.delta, best_index + coords + 1, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
			++count;
			++class_count;
		}
	}
	//printf("\n");
#ifndef GPU
	flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
#endif
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	if (count)printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);
	else printf("Count=0\n");
}
void forward_region_layer7(const region_layer l, network_state state)
{
	int coords = l.coords;
	int i, j, b, t, n;
	int size = l.coords + l.classes + 1;
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
#ifndef GPU
	flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
#endif
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w*l.n; ++i) {
			int index = size*i + b*l.outputs;
			l.output[index + coords] = logistic_activate(l.output[index + coords]);
		}
	}


#ifndef GPU
	if (l.softmax_tree) {
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
			}
		}
	}
	else if (l.softmax) {
		for (b = 0; b < l.batch; ++b) {
			for (i = 0; i < l.h*l.w*l.n; ++i) {
				int index = size*i + b*l.outputs;
				softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
			}
		}
	}
#endif
	if (!state.train) return;
	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	float avg_iou = 0, avg_hiou = 0;
	float recall = 0;
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		if (l.softmax_tree) {
			int onlyclass = 0;
			for (t = 0; t < 30; ++t) {
				box3 truth = float_to_box7(state.truth + t * (coords+1) + b*l.truths);
				if (!truth.x) break;
				int class = state.truth[t * (coords+1) + b*l.truths + coords];
				float maxp = 0;
				int maxi = 0;
				if (truth.x > 100000 && truth.y > 100000) {
					for (n = 0; n < l.n*l.w*l.h; ++n) {
						int index = size*n + b*l.outputs + coords + 1;
						float scale = l.output[index - 1];
						float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class);
						if (p > maxp) {
							maxp = p;
							maxi = n;
						}
					}
					int index = size*maxi + b*l.outputs + coords + 1;
					delta_region_class(l.output, l.delta, index, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
					++class_count;
					onlyclass = 1;
					break;
				}
			}
			if (onlyclass) continue;
		}
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
					box3 pred = get_region_box7(l.output, l.biases, n, index, i, j, l.w, l.h);
					float best_iou = 0;
					int best_class = -1;
					for (t = 0; t < 30; ++t) {
						box3 truth = float_to_box7(state.truth + t * (coords + 1) + b*l.truths);
						if (!truth.x) break;
						float iou = box_iou7(pred, truth);
						if (iou > best_iou) {
							best_class = state.truth[t * (coords + 1) + b*l.truths + coords];
							best_iou = iou;
						}
					}
					avg_anyobj += l.output[index + coords];
					//l.delta[index + coords] = l.noobject_scale * ((0 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]));
					if (l.classfix == -1) l.delta[index + coords] = l.noobject_scale * ((best_iou - l.output[index + coords]) * logistic_gradient(l.output[index + coords]));
					else {
						if (best_iou > l.thresh) {
							l.delta[index + coords] = l.object_scale * (1 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]);//0;
							if (l.classfix > 0) {
								delta_region_class(l.output, l.delta, index + coords + 1, best_class, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + coords] : 1), &avg_cat);
								++class_count;
							}
						}
						if (best_iou < 0.05) {
							l.delta[index + coords] = l.noobject_scale * (0 - l.output[index + coords]) * logistic_gradient(l.output[index + coords]);
							//do not update z and l predictions
							l.delta[index + coords - 1] = 0;
							l.delta[index + coords - 2] = 0;
						}
					}

					//not clear what this does
					if (*(state.net.seen) < 6400) {
						box3 truth = { 0 };
						truth.x = (i + .5) / l.w;
						truth.y = (j + .5) / l.h;
						truth.w = l.biases[2 * n];
						truth.h = l.biases[2 * n + 1];
						truth.rz = 0;
						truth.z = 0.5;
						truth.l = 0.5;
						if (DOABS) {
							truth.w = l.biases[2 * n] / l.w;
							truth.h = l.biases[2 * n + 1] / l.h;
						}
						delta_region_box7(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
					}
				}
			}
		}
		for (t = 0; t < 30; ++t) {
			box3 truth = float_to_box7(state.truth + t * (coords + 1) + b*l.truths);

			if (!truth.x) break;
			float best_iou = 0;
			//float rz = 0;
			int best_index = 0;
			int best_n = 0;
			i = (truth.x * l.w);
			j = (truth.y * l.h);
			//rz = truth.rz;
			//printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
			box3 truth_shift = truth;
			truth_shift.x = 0;
			truth_shift.y = 0;
			//truth_shift.rz = truth.rz;
			//printf("index %d %d\n",i, j);
			for (n = 0; n < l.n; ++n) {
				int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
				box3 pred = get_region_box7(l.output, l.biases, n, index, i, j, l.w, l.h);
				if (l.bias_match) {
					pred.w = l.biases[2 * n];
					pred.h = l.biases[2 * n + 1];
					if (DOABS) {
						pred.w = l.biases[2 * n] / l.w;
						pred.h = l.biases[2 * n + 1] / l.h;
					}
				}
				//printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
				pred.x = 0;
				pred.y = 0;
				float iou = box_iou7(pred, truth_shift);
				if (iou > best_iou) {
					best_index = index;
					best_iou = iou;
					best_n = n;
				}
			}
			//printf("%d %f (%f, %f) %f x %f, %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h, truth.rz);

			float iou = delta_region_box7(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
			box3 pred = get_region_box7(l.output, l.biases, best_n, best_index, i, j, l.w, l.h);
			float hiou = box_iou_h7(truth, pred);
			if (iou > l.thresh) recall += 1;
			avg_iou += iou;
			avg_hiou += hiou;

			avg_obj += l.output[best_index + coords];
			/*l.delta[best_index + coords] = l.object_scale * (1 - l.output[best_index + coords]) * logistic_gradient(l.output[best_index + coords]);
			if (l.rescore) {
				l.delta[best_index + coords] = l.object_scale * (iou - l.output[best_index + coords]) * logistic_gradient(l.output[best_index + coords]);
			}*/


			int class = state.truth[t * (coords + 1) + b*l.truths + coords];
			if (l.map) class = l.map[class];
			delta_region_class(l.output, l.delta, best_index + coords + 1, class, l.classes, l.softmax_tree, l.class_scale, &avg_cat);
			++count;
			++class_count;
		}
	}
	//printf("\n");
#ifndef GPU
	flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
#endif
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	if (count)printf("Region Avg IOU: %f, Class: %f, Obj: %f, HIOU: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_hiou / count, recall / count, count);
	else printf("Count=0\n");
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}
void backward_region_layer5(const region_layer l, network_state state)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}
void backward_region_layer7(const region_layer l, network_state state)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i,j,n;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}
void get_region_boxes5(layer l, int w, int h, float thresh, float **probs, box2 *boxes, int only_objectness, int *map)
{
    int i,j,n;
	int coords = l.coords;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + coords + 1) + coords;
			float scale = logistic_activate(predictions[p_index]);
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + coords + 1);
            boxes[index] = get_region_box5(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;
			//boxes[index].rz unchange

            int class_index = index * (l.classes + coords + 1) + coords + 1;
            if(l.softmax_tree){
                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}
void get_region_boxes7(layer l, int w, int h, float thresh, float **probs, box3 *boxes, int only_objectness, int *map)
{
    int i,j,n;
	int coords = l.coords;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + coords + 1) + coords;
			float scale = logistic_activate(predictions[p_index]);
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + coords + 1);
            boxes[index] = get_region_box7(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;
			//boxes[index].rz unchange

            int class_index = index * (l.classes + coords + 1) + coords + 1;
            if(l.softmax_tree){
                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    /*
       if(!state.train){
       copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
       return;
       }
     */
    flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
    if(l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(l.output_gpu+count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
            count += group_size;
        }
    }else if (l.softmax){
        softmax_gpu(l.output_gpu+5, l.classes, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 5);
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    free(cpu_state.input);
    if(!state.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
void forward_region_layer5_gpu(const region_layer l, network_state state)
{
	/*
	if(!state.train){
	copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
	return;
	}
	*/
	flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
	if (l.softmax_tree) {
		int i;
		int count = 6;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_gpu(l.output_gpu + count, group_size, l.classes + 6, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
			count += group_size;
		}
	}
	else if (l.softmax) {
		softmax_gpu(l.output_gpu + 6, l.classes, l.classes + 6, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 6);
	}

	float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.truths;
		truth_cpu = calloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
	network_state cpu_state = state;
	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;
	forward_region_layer5(l, cpu_state);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	free(cpu_state.input);
	if (!state.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer5_gpu(region_layer l, network_state state)
{
	flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
void forward_region_layer7_gpu(const region_layer l, network_state state)
{
	/*
	if(!state.train){
	copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
	return;
	}
	*/
	flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
	if (l.softmax_tree) {
		int i;
		int count = 8;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
			int group_size = l.softmax_tree->group_size[i];
			softmax_gpu(l.output_gpu + count, group_size, l.classes + 8, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
			count += group_size;
		}
	}
	else if (l.softmax) {
		softmax_gpu(l.output_gpu + 8, l.classes, l.classes + 8, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 8);
	}

	float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.truths;
		truth_cpu = calloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
	network_state cpu_state = state;
	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;
	forward_region_layer7(l, cpu_state);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	free(cpu_state.input);
	if (!state.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer7_gpu(region_layer l, network_state state)
{
	flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
#endif

