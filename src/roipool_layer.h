#include "layer.h"
#include "network.h"

#define MAX_ROI_NUMBER 30

typedef layer roipool_layer;

roipool_layer make_roipool_layer(int batch, int h, int w, int c, int n, int size, int padding);
void forward_roipool_layer(const roipool_layer l, network_state state);
void backward_roipool_layer(const roipool_layer l, network_state state);