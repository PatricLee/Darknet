#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;
typedef struct {
	float x, y, w, h, rz;
} box2;
typedef struct {
	float x, y, w, h, rz, z, l;
} box3;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f);
box2 float_to_box5(float *f);
box3 float_to_box7(float *f);
float box_iou(box a, box b);
float box_iou5(box2 a, box2 b);
float box_iou7(box3 a, box3 b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms5(box2 *boxes, float **probs, int total, int classes, float thresh);
void do_nms7(box3 *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort5(box2 *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort7(box3 *boxes, float **probs, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
