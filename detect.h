#ifndef DETECT_H
#define DETECT_H
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "box.h"

struct detection{
    box bbox;
    int classes{0};
    std::vector<float> prob;
    // std::vector<float> mask;
    float objectness{0};
    int sort_class{0};

    detection(int c): classes(c), prob(c) {}
};



// detection *make_network_boxes(int w, int h, int c, int classes, float thresh, int *num)
// {
//     int i;
//     int nboxes = w*h*c;
//     if(num) *num = nboxes;
//     detection *dets = (detection*)calloc(nboxes, sizeof(detection));
//     for(i = 0; i < nboxes; ++i){
//         dets[i].prob = (float*)calloc(classes, sizeof(float));
//     }
//     return dets;
// }

void correct_region_boxes(std::vector<detection> &dets, int n, int w, int h, int netw, int neth, int relative)
{
    // int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h))
    {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else
    {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (auto &det:dets)
    {
        box &b = det.bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative)
        {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
    }
}

static float biases[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
box get_region_box(float *x, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + x[index + 0]) / w;
    b.y = (j + x[index + 1]) / h;
    b.w = exp(x[index + 2]) * biases[2*n] / w;
    b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    return b;
}

// static int entry_index(int w, int h, int classes, int location, int entry)
// {
//     int n =   location / (w*h);
//     int loc = location % (w*h);
//     return n*w*h*(4+classes+1) + entry*w*h + loc;
// }


// void get_region_detections(float *predictions, int w, int h, int c, int classes, int w2, int h2, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
// {
//     int i, j, n;
//     for (i = 0; i < w * h; ++i)
//     {
//         int row = i / w;
//         int col = i % w;
//         for (n = 0; n < c; ++n)
//         {
//             int index = n * w * h + i;
//             for (j = 0; j < classes; ++j)
//             {
//                 dets[index].prob[j] = 0;
//             }
//             int obj_index = entry_index(w, h, classes, n * w * h + i, 4);
//             int box_index = entry_index(w, h, classes, n * w * h + i, 0);
//             // int mask_index = entry_index(w, h, classes, n * w * h + i, 4);
//             float scale = predictions[obj_index];
//             dets[index].bbox = get_region_box(predictions, n, box_index, col, row, w, h, w * h);
//             dets[index].objectness = scale > thresh ? scale : 0;
            

//             // int class_index = entry_index(w, h, classes, n * w * h + i, 4);
//             if (dets[index].objectness)
//             {
//                 for (j = 0; j < classes; ++j)
//                 {
//                     int class_index = entry_index(w, h, classes, n * w * h + i, 5 + j);
//                     float prob = scale * predictions[class_index];
//                     dets[index].prob[j] = (prob > thresh) ? prob : 0;
//                 }
//             }
//         }
//     }
//     correct_region_boxes(dets, w * h * c, w2, h2, netw, neth, relative);
// }

// void fill_network_boxes(float* predictions, int w, int h, int c, int classes, int imw, int imh, int netw, int neth, float thresh, float hier, int *map, int relative, detection *dets)
// {
//     get_region_detections(predictions, w, h, c, classes, imw, imh, netw, neth, thresh, map, hier, relative, dets);
// }

// void free_detections(detection *dets, int n)
// {
//     int i;
//     for(i = 0; i < n; ++i){
//         free(dets[i].prob);
//         if(dets[i].mask) free(dets[i].mask);
//     }
//     free(dets);
// }

// int nms_comparator(const void *pa, const void *pb)
// {
//     detection a = *(detection *)pa;
//     detection b = *(detection *)pb;
//     float diff = 0;
//     if(b.sort_class >= 0){
//         diff = a.prob[b.sort_class] - b.prob[b.sort_class];
//     } else {
//         diff = a.objectness - b.objectness;
//     }
//     if(diff < 0) return 1;
//     else if(diff > 0) return -1;
//     return 0;
// }
bool nms_comparator(const detection &a, const detection &b)
{
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    return diff > 0;
}

void do_nms_sort(std::vector<detection> &dets, int classes, float thresh)
{
    int i, j, k;
    // k = total-1;
    // for(i = 0; i <= k; ++i){
    //     if(dets[i].objectness == 0){
    //         detection swap = dets[i];
    //         dets[i] = dets[k];
    //         dets[k] = swap;
    //         --k;
    //         --i;
    //     }
    // }
    int total = dets.size();

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        std::sort(dets.begin(), dets.end(), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

#endif