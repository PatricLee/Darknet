#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "layer.h"
#include "network.h"

#define PI 3.1415926

typedef layer region_layer;
typedef layer region_layer5;
typedef layer region_layer7;

region_layer make_region_layer(int batch, int h, int w, int n, int classes, int coords);
region_layer make_region_layer5(int batch, int h, int w, int n, int classes, int coords);
region_layer make_region_layer7(int batch, int h, int w, int n, int classes, int coords);
void forward_region_layer(const region_layer l, network_state state);
void backward_region_layer(const region_layer l, network_state state);
void forward_region_layer5(const region_layer l, network_state state);
void backward_region_layer5(const region_layer l, network_state state);
void forward_region_layer7(const region_layer l, network_state state);
void backward_region_layer7(const region_layer l, network_state state);
void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void get_region_boxes5(layer l, int w, int h, float thresh, float **probs, box2 *boxes, int only_objectness, int *map);
void get_region_boxes7(layer l, int w, int h, float thresh, float **probs, box3 *boxes, int only_objectness, int *map);
void resize_region_layer(layer *l, int w, int h);
float box_iou_h7(box3 a, box3 b);

#ifdef GPU
void forward_region_layer_gpu(const region_layer l, network_state state);
void backward_region_layer_gpu(region_layer l, network_state state);
void forward_region_layer5_gpu(const region_layer l, network_state state);
void backward_region_layer5_gpu(region_layer l, network_state state);
void forward_region_layer7_gpu(const region_layer l, network_state state);
void backward_region_layer7_gpu(region_layer l, network_state state);
#endif

#endif
