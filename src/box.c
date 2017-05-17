#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "image.h"

box float_to_box(float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}
box2 float_to_box5(float *f)
{
	box2 b;
	b.x = f[0];
	b.y = f[1];
	b.w = f[2];
	b.h = f[3];
	b.rz = f[4];
	return b;
}
box3 float_to_box7(float *f)
{
	box3 b;
	b.x = f[0];
	b.y = f[1];
	b.w = f[2];
	b.h = f[3];
	b.rz = f[4];
	b.z = f[5];
	b.l = f[6];
	return b;
}

dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}
//return a score rather than iou
//now return iou via approximate method
float box_iou5(box2 a, box2 b)
{
	/*float pos, size, rot;
	pos = sqrtf(powf(a.x - b.x, 2.0) + powf(a.y - b.y, 2.0));
	pos = expf(-10.0*pos);

	size = expf(-fabsf(a.w / b.w - 1.0))*expf(-fabsf(a.h / b.h - 1.0));
	
	rot = fabsf(cosf(a.rz - b.rz));

	return pos*size*rot;*/
	//IplImage *p = cvCreateImage(cvSize(400, 400), IPL_DEPTH_8U, 1);
	float ra[9] = { cosf(a.rz), -sinf(a.rz),a.x,
					sinf(a.rz),cosf(a.rz),a.y,
					0,0,1 };
	float rb[9] = { cosf(b.rz),sinf(b.rz),-cosf(b.rz)*b.x - sinf(b.rz)*b.y,
					-sinf(b.rz),cosf(b.rz),sinf(b.rz)*b.x - cosf(b.rz)*b.y,
					0,0,1 };
	float xin[3], xmid[3], xout[3];
	/*CvMat ra, rb, x_in, x_mid, x_out;
	ra = cvMat(3, 3, CV_32FC1, ra_val);
	rb = cvMat(3, 3, CV_32FC1, rb_val);
	x_in = cvMat(3, 1, CV_32FC1, xin_val);
	x_mid = cvMat(3, 1, CV_32FC1, xmid_val);
	x_out = cvMat(3, 1, CV_32FC1, xout_val);*/
	int i, j, icount = 0;
	for (i = -5; i <= 5; i++) {
		for (j = -5; j <= 5; j++) {
			/*float xin_val[3] = { i*a.w / 9,j*a.h / 9, 1};
			cvMatMul(&ra, &x_in, &x_mid);
			cvMatMul(&rb, &x_mid, &x_out);*/
			xin[0] = i*a.w / 11;
			xin[1] = j*a.h / 11;
			xin[2] = 1;
			//printf("xin(%f,%f,%f)", xin[0], xin[1], xin[2]);
			xmid[0] = ra[0] * xin[0] + ra[1] * xin[1] + ra[2] * xin[2];
			xmid[1] = ra[3] * xin[0] + ra[4] * xin[1] + ra[5] * xin[2];
			xmid[2] = ra[6] * xin[0] + ra[7] * xin[1] + ra[8] * xin[2];
			//printf("xmid(%f,%f,%f)", xmid[0], xmid[1], xmid[2]);
			xout[0] = rb[0] * xmid[0] + rb[1] * xmid[1] + rb[2] * xmid[2];
			xout[1] = rb[3] * xmid[0] + rb[4] * xmid[1] + rb[5] * xmid[2];
			xout[2] = rb[6] * xmid[0] + rb[7] * xmid[1] + rb[8] * xmid[2];
			//printf("xout(%f,%f,%f)", xout[0], xout[1], xout[2]);
			if (xout[0]<(b.w / 2) && xout[0]>(-b.w / 2) && xout[1]<(b.h / 2) && xout[1]>(-b.h / 2)) {
				icount++;
			}
		}
	}
	//cvReleaseMat(&ra);
	//cvReleaseMat(&rb);
	//cvReleaseMat(&x);
	float inter = a.w*a.h / 121 * icount;
	float uni = a.w*a.h + b.w*b.h - inter;
	return inter / uni;

	/*float i, u;
	float a_l, a_r, a_t, a_b, b_l, b_r, b_t, b_b;
	float i_l, i_r, i_t, i_b;
	float boarder_temp1, boarder_temp2;
	boarder_temp1 = -a.w / 2 * cosf(a.rz) - a.h / 2 * sinf(a.rz);
	boarder_temp2 = -a.w / 2 * cosf(a.rz) + a.h / 2 * cosf(a.rz);
	a_l = (boarder_temp1 < boarder_temp2) ? boarder_temp1 : boarder_temp2;
	a_l += a.x;
	a_l = (a_l > 0) ? a_l : 0;
	boarder_temp1 = a.w / 2 * cosf(a.rz) - a.h / 2 * sinf(a.rz);
	boarder_temp2 = a.w / 2 * cosf(a.rz) + a.h / 2 * sinf(a.rz);
	a_r = (boarder_temp1 > boarder_temp2) ? boarder_temp1 : boarder_temp2;
	a_r += a.x;
	a_r = (a_r < 1) ? a_r : 1;
	boarder_temp1 = a.w / 2 * sinf(a.rz) - a.h / 2 * cosf(a.rz);
	boarder_temp2 = -a.w / 2 * sinf(a.rz) - a.h / 2 * cosf(a.rz);
	a_t = (boarder_temp1 < boarder_temp2) ? boarder_temp1 : boarder_temp2;
	a_t += a.y;
	a_t = (a_t > 0) ? a_t : 0;
	boarder_temp1 = a.w / 2 * sinf(a.rz) + a.h / 2 * cosf(a.rz);
	boarder_temp2 = -a.w / 2 * sinf(a.rz) + a.h / 2 * cosf(a.rz);
	a_b = (boarder_temp1 > boarder_temp2) ? boarder_temp1 : boarder_temp2;
	a_b += a.y;
	a_b = (a_b < 1) ? a_b : 1;

	boarder_temp1 = -b.w / 2 * cosf(b.rz) - b.h / 2 * sinf(b.rz);
	boarder_temp2 = -b.w / 2 * cosf(b.rz) + b.h / 2 * cosf(b.rz);
	b_l = (boarder_temp1 < boarder_temp2) ? boarder_temp1 : boarder_temp2;
	b_l += b.x;
	b_l = (b_l > 0) ? b_l : 0;
	boarder_temp1 = b.w / 2 * cosf(b.rz) - b.h / 2 * sinf(b.rz);
	boarder_temp2 = b.w / 2 * cosf(b.rz) + b.h / 2 * sinf(b.rz);
	b_r = (boarder_temp1 > boarder_temp2) ? boarder_temp1 : boarder_temp2;
	b_r += b.x;
	b_r = (b_r < 1) ? b_r : 1;
	boarder_temp1 = b.w / 2 * sinf(b.rz) - b.h / 2 * cosf(b.rz);
	boarder_temp2 = -b.w / 2 * sinf(b.rz) - b.h / 2 * cosf(b.rz);
	b_t = (boarder_temp1 < boarder_temp2) ? boarder_temp1 : boarder_temp2;
	b_t += b.y;
	b_t = (b_t > 0) ? b_t : 0;
	boarder_temp1 = b.w / 2 * sinf(b.rz) + b.h / 2 * cosf(b.rz);
	boarder_temp2 = -b.w / 2 * sinf(b.rz) + b.h / 2 * cosf(b.rz);
	b_b = (boarder_temp1 > boarder_temp2) ? boarder_temp1 : boarder_temp2;
	b_b += b.y;
	b_b = (b_b < 1) ? b_b : 1;

	i_l = (a_l > b_l) ? a_l : b_l;
	i_r = (a_r < b_r) ? a_r : b_r;
	i_t = (a_t > b_t) ? a_t : b_t;
	i_b = (a_b < b_b) ? a_b : b_b;

	if (i_l >= i_r || i_t >= i_b)return 0;
	i = (i_r - i_l)*(i_b - i_t);

	u = (a_r - a_l)*(a_b - a_t) + (b_r - b_l)*(b_b - b_t) - i;

	return i / u;*/
}
float box_iou7(box3 a, box3 b){
	return box_iou5(*((box2 *)&a), *((box2 *)&b));
}

float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.h-b.h, 2));
}

dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}

typedef struct{
    int index;
    int class;
    float **probs;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}
void do_nms_sort5(box2 *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

	for (i = 0; i < total; ++i) {
		s[i].index = i;
		s[i].class = 0;
		s[i].probs = probs;
	}

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			s[i].class = k;
		}
		qsort(s, total, sizeof(sortable_bbox), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (probs[s[i].index][k] == 0) continue;
			box2 a = boxes[s[i].index];
			for (j = i + 1; j < total; ++j) {
				box2 b = boxes[s[j].index];
				if (box_iou5(a, b) > thresh) {
					probs[s[j].index][k] = 0;
				}
			}
		}
	}
	free(s);
}void do_nms_sort7(box3 *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

	for (i = 0; i < total; ++i) {
		s[i].index = i;
		s[i].class = 0;
		s[i].probs = probs;
	}

	for (k = 0; k < classes; ++k) {
		for (i = 0; i < total; ++i) {
			s[i].class = k;
		}
		qsort(s, total, sizeof(sortable_bbox), nms_comparator);
		for (i = 0; i < total; ++i) {
			if (probs[s[i].index][k] == 0) continue;
			box3 a = boxes[s[i].index];
			for (j = i + 1; j < total; ++j) {
				box3 b = boxes[s[j].index];
				if (box_iou7(a, b) > thresh) {
					probs[s[j].index][k] = 0;
				}
			}
		}
	}
	free(s);
}

void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}
void do_nms5(box2 *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	for (i = 0; i < total; ++i) {
		int any = 0;
		for (k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
		if (!any) {
			continue;
		}
		for (j = i + 1; j < total; ++j) {
			if (box_iou5(boxes[i], boxes[j]) > thresh) {
				for (k = 0; k < classes; ++k) {
					if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
					else probs[j][k] = 0;
				}
			}
		}
	}
}
void do_nms7(box3 *boxes, float **probs, int total, int classes, float thresh)
{
	int i, j, k;
	for (i = 0; i < total; ++i) {
		int any = 0;
		for (k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
		if (!any) {
			continue;
		}
		for (j = i + 1; j < total; ++j) {
			if (box_iou7(boxes[i], boxes[j]) > thresh) {
				for (k = 0; k < classes; ++k) {
					if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
					else probs[j][k] = 0;
				}
			}
		}
	}
}

box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
