#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "cronus.h"
#include "overseer.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

roi_boxes *roiboxes;

void cronus2_trainmain(char *datacfg, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA3;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_cronus7(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_cronus7(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
		
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
			if (overseer.level) {
				float result[6];
				cronus_recall5_online(net, datacfg, VAL_TRAINVAL, &result);
				OverseerPushPoints(i, result[0], result[1], result[2], result[3],
					result[4], result[5], result[6]);
			}
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
void cronus2_trainsub(char *datacfg, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config2", "cronus.cfg");
    char *train_images = option_find_str(options, "train2", "data/train.list");
    char *backup_directory = option_find_str(options, "backup2", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA0;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_cronus(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_cronus(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network_cronus(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
			if (overseer.level) {
				float result[6];
				cronus_recall5_online(net, datacfg, VAL_TRAINVAL, &result);
				OverseerPushPoints(i, result[0], result[1], result[2], result[3],
					result[4], result[5], result[6]);
			}
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}
void cronus_train(char *datacfg, char *weightfile, int *gpus, int ngpus, int clear)
{
    list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network *nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if(clear) *nets[i].seen = 0;
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA2;
    args.threads = 8;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_cronus(args);
    clock_t time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
		//cronus does not resize
		/*if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+100 > net.max_batches) dim = 544;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets + i, dim, dim);
            }
            net = nets[0];
        }*/
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_cronus(args);

        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
           image im = float_to_image(448, 448, 3, train.X.vals[10]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
           draw_bbox(im, b, 8, 1,0,0);
           }
           save_image(im, "truth11");
         */

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network_cronus(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
			if (overseer.level) {
				float result[6];
				cronus_recall5_online(net, datacfg, VAL_TRAINVAL, &result);
				OverseerPushPoints(i, result[0], result[1], result[2], result[3],
					result[4], result[5], result[6]);
			}
        }
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

//offline testing, draw boxes, show and save image
void cronus_test5(char *datacfg, char *weightfile, float thresh, int validate_type)
{
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	network net = parse_network_cfg(cfgfile);

	image **alphabet = load_alphabet();
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = NMS_THRESH;
	layer l = net.layers[net.n - 1];
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *test_images = option_find_str(options, "train", "data/test.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			//image sized = resize_image(im, net.w, net.h);
			box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = im.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort5(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections5(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1)", input);
			find_replace(input, ".png", "(1)", input);
			show_image(im, "Cronus");
			cvWaitKey(50);
			save_image_png(im, input);
			free_image(im);
			//free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *test_images = option_find_str(options, "valid", "data/val.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			image sized = resize_image(im, net.w, net.h);
			box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = sized.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort5(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections5(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1).jpg", input);
			find_replace(input, ".png", "(1).png", input);
			show_image(im, "Cronus");
			cvWaitKey(50);
			save_image(im, input);
			free_image(im);
			free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
}
void cronus2_testmain(char *datacfg, char *weightfile, float thresh, int validate_type)
{
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	network net = parse_network_cfg(cfgfile);

	image **alphabet = load_alphabet();
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = NMS_THRESH;
	layer l = net.layers[net.n - 1];
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *test_images = option_find_str(options, "train", "data/test.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			//image sized = resize_image(im, net.w, net.h);
			box3 *boxes = calloc(l.w*l.h*l.n, sizeof(box3));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = im.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes7(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort7(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections7(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1)", input);
			find_replace(input, ".png", "(1)", input);
			show_image(im, "Cronus");
			cvWaitKey(50);
			save_image_png(im, input);
			free_image(im);
			//free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *test_images = option_find_str(options, "valid", "data/val.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			image sized = resize_image(im, net.w, net.h);
			box3 *boxes = calloc(l.w*l.h*l.n, sizeof(box3));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = sized.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes7(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort7(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections7(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1).jpg", input);
			find_replace(input, ".png", "(1).png", input);
			show_image(im, "Cronus");
			cvWaitKey(50);
			save_image(im, input);
			free_image(im);
			free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
}
void cronus2_testsub(char *datacfg, char *weightfile, float thresh, int validate_type)
{
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config2", "cronus.cfg");
	char *name_list = option_find_str(options, "names2", "data/names.list");
	char **names = get_labels(name_list);
	network net = parse_network_cfg(cfgfile);

	image **alphabet = load_alphabet();
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	clock_t time;
	char buff[256];
	char *input = buff;
	int j;
	float nms = NMS_THRESH;
	layer l = net.layers[net.n - 1];
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *test_images = option_find_str(options, "train2", "data/test.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			//image sized = resize_image(im, net.w, net.h);
			box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = im.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort5(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections5(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1)", input);
			find_replace(input, ".png", "(1)", input);
			show_image(im, "Cronus Sub");
			cvWaitKey(50);
			save_image_png(im, input);
			free_image(im);
			//free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *test_images = option_find_str(options, "valid2", "data/val.list");
		list *plist = get_paths(test_images);
		char **paths = (char **)list_to_array(plist);
		
		for (int n = 0; n < plist->size; n++) {
			strncpy(input, paths[n], 256);
			image im = load_image_color(input, 0, 0);
			image sized = resize_image(im, net.w, net.h);
			box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
			float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
			for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
			float *X = sized.data;
			time = clock();
			network_predict(net, X);
			printf("%d: Predicted in %f seconds.\n", n, sec(clock() - time));
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 0, 0);
			if (nms) do_nms_sort5(boxes, probs, l.w*l.h*l.n, l.classes, nms);
			draw_detections5(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
			//save_image(im, "predictions");
			find_replace(input, ".jpg", "(1).jpg", input);
			find_replace(input, ".png", "(1).png", input);
			show_image(im, "Cronus Sub");
			cvWaitKey(50);
			save_image(im, input);
			free_image(im);
			free_image(sized);
			free(boxes);
			free_ptrs((void **)probs, l.w*l.h*l.n);
		}
	}
}
//for a certain file input
void cronus_test5_single(char *datacfg, char *weightfile, float thresh, char* filename)
{
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	network net = parse_network_cfg(cfgfile);

	image **alphabet = load_alphabet();
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	clock_t time;
	char *input = filename;
	int j;
	float nms = NMS_THRESH;
	layer l = net.layers[net.n - 1];
	image im = load_image_color(input, 0, 0);
	image sized = resize_image(im, net.w, net.h);
	box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
	float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
	float *X = sized.data;
	time = clock();
	network_predict(net, X);
	printf("Predicted in %f seconds.\n", sec(clock() - time));
	get_region_boxes5(l, 1, 1, thresh, probs, boxes, 0, 0);
	if (nms) do_nms_sort5(boxes, probs, l.w*l.h*l.n, l.classes, nms);
	draw_detections5(&im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
	find_replace(input, ".jpg", "(1).jpg", input);
	find_replace(input, ".png", "(1).png", input);
	show_image(im, "Cronus");
	cvWaitKey(0);
	save_image(im, input);
	free_image(im);
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
//#ifdef OPENCV
//	Mat showpic = imread(input);
//	imshow("detection", showpic);
//	waitKey(0);
//#endif
}
//online validating, does not show or draw boxes
void cronus_recall5_online(network net, char* datacfg, int validate_type, float *result)
{
	int j, k;
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char *prefix = option_find_str(options, "results", "results");
	char **names = get_labels(name_list);
	
	int batches_t = net.batch;
	set_batch_network(&net, 1);
	srand(time(0));
	
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *valid_images = option_find_str(options, "train", "data/train.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);
		
		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, sized.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		if (result) {
			result[0] = avg_iou * 100 / total;
			result[1] = (float)proposals / (i + 1);
			result[2] = 100.*correct / total;
		}
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *valid_images = option_find_str(options, "valid", "data/valid.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);
		
		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, sized.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			free_image(sized);
		}
		fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		if (result) {
			result[3] = avg_iou * 100 / total;
			result[4] = (float)proposals / (i + 1);
			result[5] = 100.*correct / total;
		}
	}
	//network net = parse_network_cfg(cfgfile);
	/*if (weightfile) {
		load_weights(&net, weightfile);
	}*/
	
	//fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	set_batch_network(&net, batches_t);
	
	return;
}
//offline recall, just numbers
void cronus2_recallmain(char* datacfg, char* weights, int validate_type)
{
	int j, k;
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	network net = parse_network_cfg(cfgfile);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();
	if (weights) {
		load_weights(&net, weights);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *valid_images = option_find_str(options, "train", "data/train.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);

		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box3 *boxes = calloc(l.w*l.h*l.n, sizeof(box3));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes7(l, 1, 1, thresh, probs, boxes, 1, 0);
			
			if (nms) do_nms7(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label3 *truth = read_boxes7(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box3 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz, truth[j].z, truth[j].l};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou7(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *valid_images = option_find_str(options, "valid", "data/valid.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);
		
		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box3 *boxes = calloc(l.w*l.h*l.n, sizeof(box3));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes7(l, 1, 1, thresh, probs, boxes, 1, 0);
			if (nms) do_nms7(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label3 *truth = read_boxes7(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box3 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz, truth[j].z, truth[j].l};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou7(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	//network net = parse_network_cfg(cfgfile);
	/*if (weightfile) {
		load_weights(&net, weightfile);
	}*/
	
	//fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	//set_batch_network(&net, batches_t);
	
	return;
}
void cronus2_recallsub(char* datacfg, char* weights, int validate_type)
{
	int j, k;
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	network net = parse_network_cfg(cfgfile);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();
	if (weights) {
		load_weights(&net, weights);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *valid_images = option_find_str(options, "train", "data/train.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);

		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *valid_images = option_find_str(options, "valid", "data/valid.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);
		
		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	//network net = parse_network_cfg(cfgfile);
	/*if (weightfile) {
		load_weights(&net, weightfile);
	}*/
	
	//fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	//set_batch_network(&net, batches_t);
	
	return;
}
void cronus_recall5(char* datacfg, char* weights, int validate_type)
{
	int j, k;
	list *options = read_data_cfg(datacfg);
	char *cfgfile = option_find_str(options, "config", "cronus.cfg");
	network net = parse_network_cfg(cfgfile);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);
	image **alphabet = load_alphabet();
	if (weights) {
		load_weights(&net, weights);
	}
	set_batch_network(&net, 1);
	srand(time(0));
	
	if(validate_type & VAL_TRAIN){
		printf("validating on training set\n");
		
		char *valid_images = option_find_str(options, "train", "data/train.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);

		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	if(validate_type & VAL_VAL){
		printf("validating on validating set\n");
		
		char *valid_images = option_find_str(options, "valid", "data/valid.list");
		list *plist = get_paths(valid_images);
		char **paths = (char **)list_to_array(plist);
		
		layer l = net.layers[net.n - 1];
		int classes = l.classes;
		box2 *boxes = calloc(l.w*l.h*l.n, sizeof(box2));
		float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
		for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));
		int m = plist->size;
		int i = 0;
		
		float thresh = RP_THRESH;
		float iou_thresh = l.thresh;
		float nms = NMS_THRESH;
		int total = 0;
		int correct = 0;
		int proposals = 0;
		float avg_iou = 0;
		for (i = 0; i < m; ++i) {
			char *path = paths[i];
			image orig = load_image_color(path, 0, 0);
			//image sized = resize_image(orig, net.w, net.h);
			char *id = basecfg(path);
			network_predict(net, orig.data);
			get_region_boxes5(l, 1, 1, thresh, probs, boxes, 1, 0);
			if (nms) do_nms5(boxes, probs, l.w*l.h*l.n, 1, nms);
			char labelpath[4096];
			find_replace(path, ".jpg", ".txt", labelpath);
			find_replace(labelpath, ".png", ".txt", labelpath);
			int num_labels = 0;
			box_label2 *truth = read_boxes5(labelpath, &num_labels);
			for (k = 0; k < l.w*l.h*l.n; ++k) {
				if (probs[k][0] > thresh) {
					++proposals;
				}
			}
			for (j = 0; j < num_labels; ++j) {
				++total;
				box2 t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].rz};
				float best_iou = 0;
				for (k = 0; k < l.w*l.h*l.n; ++k) {
					float iou = box_iou5(boxes[k], t);
					if (probs[k][0] > thresh && iou > best_iou) {
						best_iou = iou;
					}
				}
				avg_iou += best_iou;
				if (best_iou > iou_thresh) {
					++correct;
				}
			}
			free(id);
			free_image(orig);
			//free_image(sized);
		}
		fprintf(stderr, "%5d correct out of %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
	}
	//network net = parse_network_cfg(cfgfile);
	/*if (weightfile) {
		load_weights(&net, weightfile);
	}*/
	
	//fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	//set_batch_network(&net, batches_t);
	
	return;
}

void run_cronus(int argc, char **argv)
{
	if (find_arg(argc, argv, "-h") == 1) {
		//show help

		return;
	}
	if(argc < 3){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
	char *datacfg = argv[3];
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	

    //char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    //int frame_skip = find_int_arg(argc, argv, "-s", 0);
    
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
	
	char *weights = (argc > 4) ? argv[4] : 0;
	int val_type = 0;
	char *filename=calloc(128,sizeof(char));
	if(argc>5){
		if(0==strcmp(argv[5],"train"))val_type=VAL_TRAIN;
		else if(0==strcmp(argv[5],"val"))val_type=VAL_VAL;
		else if(0==strcmp(argv[5],"trainval"))val_type=VAL_TRAINVAL;
		else{
			strcpy(filename, argv[5]);
		}
	}
	
	if (0 == strcmp(argv[2], "train")) {
		//cronus train datafile (weights)
		int level = option_find_int(options, "overseer", 0);
		char *overseer_path = option_find_str(options, "overseer_path", "overseer.txt");
		char para[128];
		*((int *)para)=level;
		strcpy(para + sizeof(int), overseer_path);
		pthread_t overseer_init;
		if (level)pthread_create(&overseer_init, 0, OverseerInitialize, para);
		cronus_train(datacfg, weights, gpus, ngpus, clear);
	}
	else if (0 == strcmp(argv[2], "recall")) {
		//cronus recall datafile weights trainval
		if(weights == NULL || val_type == 0) return;
		cronus_recall5(datacfg, weights, val_type);
	}
	else if (0 == strcmp(argv[2], "online")) {
		//online_cronus();
	}
	else if (0 == strcmp(argv[2], "test")) {
		//cronus test datafile weights trainval
		//cronus test datafile weights filename
		if(val_type)cronus_test5(datacfg, weights, thresh, val_type);
		else if(filename)cronus_test5_single(datacfg, weights, thresh, filename);
	}
	else {
		//no match

		return;
	}
}
void run_cronus2(int argc, char **argv)
{
	if (find_arg(argc, argv, "-h") == 1) {
		//show help

		return;
	}
	if(argc < 3){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
	char *datacfg = argv[3];
	list *options = read_data_cfg(datacfg);
	char *name_list = option_find_str(options, "names", "data/names.list");
	

    //char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    //int frame_skip = find_int_arg(argc, argv, "-s", 0);
    
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
	
	char *weights = (argc > 4) ? argv[4] : 0;
	int val_type = 0;
	char *filename=calloc(128,sizeof(char));
	if(argc>5){
		if(0==strcmp(argv[5],"train"))val_type=VAL_TRAIN;
		else if(0==strcmp(argv[5],"val"))val_type=VAL_VAL;
		else if(0==strcmp(argv[5],"trainval"))val_type=VAL_TRAINVAL;
		else{
			strcpy(filename, argv[5]);
		}
	}
	
	if (0 == strcmp(argv[2], "trainmain")) {
		//cronus train datafile (weights)
		/*int level = option_find_int(options, "overseer", 0);
		char *overseer_path = option_find_str(options, "overseer_path", "overseer.txt");
		char para[128];
		*((int *)para)=level;
		strcpy(para + sizeof(int), overseer_path);
		pthread_t overseer_init;
		if (level)pthread_create(&overseer_init, 0, OverseerInitialize, para);*/
		cronus2_trainmain(datacfg, weights, gpus, ngpus, clear);
	}
	if (0 == strcmp(argv[2], "trainsub")) {
		//cronus train datafile (weights)
		int level = option_find_int(options, "overseer", 0);
		char *overseer_path = option_find_str(options, "overseer_path", "overseer.txt");
		char para[128];
		*((int *)para) = level;
		strcpy(para + sizeof(int), overseer_path);
		pthread_t overseer_init;
		if (level)pthread_create(&overseer_init, 0, OverseerInitialize, para);
		cronus2_trainsub(datacfg, weights, gpus, ngpus, clear);
	}
	else if (0 == strcmp(argv[2], "recallmain")) {
		//cronus recall datafile weights trainval
		if(weights == NULL || val_type == 0) return;
		cronus2_recallmain(datacfg, weights, val_type);
	}
	else if (0 == strcmp(argv[2], "recallsub")) {
		//cronus recall datafile weights trainval
		if(weights == NULL || val_type == 0) return;
		cronus2_recallsub(datacfg, weights, val_type);
	}
	else if (0 == strcmp(argv[2], "online")) {
		//online_cronus();
	}
	else if (0 == strcmp(argv[2], "testmain")) {
		//cronus test datafile weights trainval
		//cronus test datafile weights filename
		if(val_type)cronus2_testmain(datacfg, weights, thresh, val_type);
		//else if(filename)cronus_test5_single(datacfg, weights, thresh, filename);
	}
	else if (0 == strcmp(argv[2], "testsub")) {
		//cronus test datafile weights trainval
		//cronus test datafile weights filename
		if(val_type)cronus2_testsub(datacfg, weights, thresh, val_type);
		//else if(filename)cronus_test5_single(datacfg, weights, thresh, filename);
	}
	else {
		//no match

		return;
	}
}
