#include <cstdlib>
#include <systemc.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <time.h>
#include <sys/time.h>
#include "kahn_process.h"
#include "cfg.h"
#include "detect.h"
#include "image.h"

class ConvLayer : public kahn_process
{
public:
    const int w;
    const int h;
    const int n;
    const int c;
    const int s;
    const int size;
    const int pad;
    const int out_w;
    const int out_h;
    const bool leaky;
    const bool batchnorm;

    std::vector<float> weight;
    std::vector<float> mean;
    std::vector<float> variance;
    std::vector<float> scale;
    std::vector<float> bias;
    std::vector<float> buffer;

    sc_fifo_in<float> weight_in;
    sc_fifo_in<float> act_in;
    sc_fifo_out<float> act_out;
    sc_fifo_in<bool> ctrl_in;
    sc_fifo_out<bool> ctrl_out;

    ConvLayer(sc_module_name name,
              int _w,
              int _h,
              int _n,
              int _c,
              int _s,
              int _size,
              int _pad,
              bool _leaky,
              bool _bn) : kahn_process(name), w(_w), h(_h), n(_n), c(_c), s(_s), size(_size), pad(_pad),
                             out_w((_w - _size) / _s + 1), out_h((_h - _size) / _s + 1), leaky(_leaky), batchnorm(_bn),
                             weight(_n * _c * _size * _size),
                             mean(_n),
                             variance(_n),
                             scale(_n),
                             bias(_n),
                             buffer(_c * _size * _w)
    {
    }

    void process() override
    {
        bool ctrl;
        ctrl_in->read(ctrl);
        ctrl_out->write(ctrl);
        // fprintf(stderr, "Start %s\n", name());
        for (int i = 0; i != n; ++i)
            weight_in->read(bias[i]);
        if (batchnorm){
            for (int i = 0; i != n; ++i)
                weight_in->read(scale[i]);
            for (int i = 0; i != n; ++i)
                weight_in->read(mean[i]);
            for (int i = 0; i != n; ++i)
                weight_in->read(variance[i]);
        }
        for (int i = 0; i != n * c * size * size; ++i)
            weight_in->read(weight[i]);
        // fprintf(stderr, "%s weight loaded.\n", name());


        for (auto &i : buffer)
            i = 0.0;
        std::vector<float> result(n);

        int cur = 0;
        for (int i = 0; i < h+pad; ++i)
        {
            // fprintf(stderr, "%s i=%d.\n", name(),i);
            for (int j = 0; j < w+pad; ++j)
            {
                if (i<h && j<w)
                    for (int k = 0; k != c; ++k)
                        act_in->read(buffer[(cur * w + j) * c + k]);

                if (i>=pad && j>=pad && (i-pad) % s == 0 && (j-pad) % s == 0)
                {
                    for (auto &x : result)
                        x = 0.0;
                    for (int k = 0; k != c; ++k)
                        for (int y = 0; y != size; ++y)
                            for (int x = 0; x != size; ++x) {
                                int x1 = j + x - 2*pad;
                                int y1 = i + y - 2*pad;
                                if (x1 >= 0 && y1 >= 0 && x1 < w && y1 < h)
                                    for (int m = 0; m != n; ++m)
                                    {
                                        // if (strcmp(name(), "kpn_yolo.Conv0")==0 && i==0 && j==0 && m==0)
                                            // printf("%d %d %d %.4f %.4f\n", x1, y1, k, buffer[((y1 % size) * w + x1) * c + k], weight[m * c * size * size + k * size * size + y * size + x]);
                                        result[m] += buffer[((y1 % size) * w + x1) * c + k] * weight[m * c * size * size + k * size * size + y * size + x];
                                    }
                            }
                    if (batchnorm)
                        for (int m = 0; m != n; ++m)
                            result[m] = (result[m] - mean[m]) / (sqrt(variance[m]) + .000001f) * scale[m] + bias[m];
                    else
                        for (int m = 0; m != n; ++m)
                            result[m] = result[m] + bias[m];
                    if (leaky)
                        for (auto &m : result)
                            m = (m > 0) ? m : .1 * m;
                    for (int m = 0; m != n; ++m)
                        act_out->write(result[m]);
                }
            }
            cur = (cur + 1) % size;
        }
    }
};

class MaxPoolingLayer : public kahn_process
{
public:
    const int w;
    const int h;
    const int c;
    const int s;
    const int size;
    const int out_w;
    const int out_h;

    std::vector<float> buffer;

    sc_fifo_in<float> act_in;
    sc_fifo_out<float> act_out;
    sc_fifo_in<bool> ctrl_in;
    sc_fifo_out<bool> ctrl_out;

    MaxPoolingLayer(sc_module_name name, int _w, int _h, int _c, int _s, int _size) : kahn_process(name), w(_w), h(_h), c(_c), s(_s), size(_size),
                                                                                      out_w(w / size), out_h(h / size),
                                                                                      buffer(_size * _w * _c)
    {
    }

    void process() override
    {
        bool ctrl;
        ctrl_in->read(ctrl);
        ctrl_out->write(ctrl);
        // fprintf(stderr, "Start %s\n", name());
        std::vector<float> result(c);
        int cur = 0;
        int pad =size/2;
        for (int i = 0; i < h+pad; i++)
        {
            for (int j = 0; j < w+pad; ++j)
            {
                if (i<h && j<w)
                    for (int k = 0; k != c; ++k)
                        act_in->read(buffer[(cur * w + j) * c + k]);
                if (i>=pad && j>=pad && (i-pad) % s == 0 && (j-pad) % s == 0)
                {
                    for (auto &x : result)
                        x = -__FLT_MAX__;
                    for (int x = 0; x != size; ++x)
                    {
                        for (int y = 0; y != size; ++y)
                        {
                            for (int k = 0; k != c; ++k)
                            {
                                int x1 = j + x - pad;
                                int y1 = i + y - pad;
                                if (x1 >= 0 && y1 >= 0 && x1 < w && y1 < h)
                                    result[k] = std::max(result[k], buffer[(y1 % size * w + x1) * c + k]);
                            }
                        }
                    }
                    for (int k = 0; k != c; ++k)
                        act_out->write(result[k]);
                }
            }
            cur = (cur + 1) % size;
        }
    }
};

class RegionLayer: public kahn_process{
public:
    const std::string filename;
    const int w;
    const int h;
    const int c;
    const int netw;
    const int neth;

    sc_fifo_in<bool> ctrl_in;
    sc_fifo_in<float> act_in;

    std::vector<float> buffer;

    RegionLayer(sc_module_name name, const char* fname, int _w, int _h, int _c, int _netw, int _neth):
        kahn_process(name), filename(fname), w(_w), h(_w), c(_c), netw(_netw), neth(_neth), buffer(_c){}


    void process() override{
        bool ctrl;
        ctrl_in.read(ctrl);
        // fprintf(stderr, "Start %s\n", name());
        char **names = get_labels("data/coco.names");
        image **alphabet = load_alphabet();

        int classes = 80;
        image im = load_image_color(filename.c_str(),0,0);
        double t0 = what_time_is_it_now();
        std::vector<detection> dets;


        for (int i=0;i!=h;++i)
            for (int j=0;j!=w;++j) {
                for (int k=0;k!=c;++k)
                    act_in.read(buffer[k]);
                for (int n=0;n<5;++n){
                    buffer[n*(classes+5)] = 1./(1. + exp(-buffer[n*(classes+5)]));
                    buffer[n*(classes+5)+1] = 1./(1. + exp(-buffer[n*(classes+5)+1]));
                    buffer[n*(classes+5)+4] = 1./(1. + exp(-buffer[n*(classes+5)+4]));
                }
                for (int b=0;b<5;++b){
                    softmax(buffer.data()+b*(classes+5)+5, classes, 1, buffer.data()+b*(classes+5)+5);
                }

                for (int n=0;n<5;++n){
                    // int index = n * w * h + i*w + j;
                    detection det(classes);
                    for (int x = 0; x < classes; ++x)
                        det.prob[x] = 0;
                    float scale = buffer[n*(classes+5)+4];
                    det.bbox = get_region_box(buffer.data(), n, n*(classes+5), j, i, w, h);
                    det.objectness = scale > 0.5 ? scale : 0;
                    if (det.objectness) {
                        for (int x = 0; x < classes; ++x)
                        {
                            float prob = scale * buffer[n*(classes+5)+5+x];
                            det.prob[x] = (prob > .5) ? prob : 0;
                        }
                        dets.push_back(std::move(det));
                    }
                }
            }

        correct_region_boxes(dets, w * h * 5, im.w, im.h, netw, neth, 1);
        
        printf("%s: Predicted in %f seconds.\n", filename.c_str(), what_time_is_it_now()-t0);

        // fill_network_boxes(buffer2.data(), w, h, 5, classes, im.w, im.h, netw, neth, 0.5, 0.5, 0, 1, dets);
        float nms=.45;
        do_nms_sort(dets, classes, nms);
        draw_detections(im, dets, 0.5, names, alphabet, classes);
        // free_detections(dets, nboxes);
        save_image(im, "predictions");
        free_image(im);
    }
    void softmax(float *input, int n, int stride, float *output)
    {
        int i;
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < n; ++i){
            if(input[i*stride] > largest) largest = input[i*stride];
        }
        for(i = 0; i < n; ++i){
            float e = exp(input[i*stride] - largest);
            sum += e;
            output[i*stride] = e;
        }
        for(i = 0; i < n; ++i){
            output[i*stride] /= sum;
        }
    }

    double what_time_is_it_now()
    {
        struct timeval time;
        if (gettimeofday(&time,NULL)){
            return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

};

class WeightLoader: public kahn_process{
public:
    std::string cfg_file;
    std::string weight_file;
    std::vector<sc_fifo_out<float>> weight_out;

    std::vector<int> weight_size;
    std::vector<int> channels;

    sc_fifo_in<bool> ctrl_in;
    sc_fifo_out<bool> ctrl_out;

    WeightLoader(sc_module_name name, const char* cfg, const char* weight, int numWeight): kahn_process(name), cfg_file(cfg), weight_file(weight), weight_out(numWeight)
    {}

    void parse_network_cfg()
    {
        list *sections = read_cfg(cfg_file.c_str());
        node *n = sections->front;
        size_params params;

        section *s = (section *)n->val;
        list *options = s->options;
        params.h = option_find_int_quiet(options, "height",0);
        params.w = option_find_int_quiet(options, "width",0);
        params.c = option_find_int_quiet(options, "channels",0);

        params.inputs = params.h*params.w*params.c;
        params.batch = 1;

        // size_t workspace_size = 0;
        n = n->next;
        int count = 0;
        free_section(s);
        fprintf(stderr, "layer     filters    size              input                output\n");
        while(n){
            params.index = count;
            fprintf(stderr, "%5d ", count);
            s = (section *)n->val;
            options = s->options;
            LAYER_TYPE lt = string_to_layer_type(s->type);
            if(lt == CONVOLUTIONAL){
                int n = option_find_int(options, "filters",1);
                int size = option_find_int(options, "size",1);
                int stride = option_find_int(options, "stride",1);
                int pad = option_find_int_quiet(options, "pad",0);
                int padding = option_find_int_quiet(options, "padding",0);
                int groups = option_find_int_quiet(options, "groups", 1);
                if(pad) padding = size/2;

                // char *activation_s = option_find_str(options, "activation", "logistic");
                // ACTIVATION activation = get_activation(activation_s);

                int h,w,c;
                h = params.h;
                w = params.w;
                c = params.c;
                // batch=params.batch;

                // int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

                channels.push_back(n);
                weight_size.push_back(size*size*c*n);
                
                params.h = (h+2*padding-size)/stride+1;
                params.w = (w+2*padding-size)/stride+1;
                params.c = n;

                fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, params.w, params.h, params.c, (2.0 * n * size*size*c/groups * params.h*params.w)/1000000000.);
            }else if(lt == REGION){
                fprintf(stderr, "detection\n");
                srand(0);
            }else if(lt == MAXPOOL){
                int stride = option_find_int(options, "stride",1);
                int size = option_find_int(options, "size",stride);
                int padding = option_find_int_quiet(options, "padding", size-1);

                int h,w,c;
                h = params.h;
                w = params.w;
                c = params.c;
                // batch=params.batch;

                params.w = (w + padding - size)/stride + 1;
                params.h = (h + padding - size)/stride + 1;
                params.c = c;
                fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", size, size, stride, w, h, c, params.w, params.h, params.c);
                // l = parse_maxpool(options, params);
            }
            free_section(s);
            n = n->next;
            ++count;
        }
        free_list(sections);
        fprintf(stderr, "mask_scale: Using default '1.000000'\n");
    }

    void process() override{
        bool ctrl;
        ctrl_in.read(ctrl);
        ctrl_out.write(ctrl);
        // fprintf(stderr, "Start weight loader\n");
        parse_network_cfg();
        std::fprintf(stderr, "Loading weights from %s...", weight_file.c_str());
        std::fflush(stdout);
        FILE *fp = fopen(weight_file.c_str(), "rb");

        int major;
        int minor;
        int revision;
        std::fread(&major, sizeof(int), 1, fp);
        std::fread(&minor, sizeof(int), 1, fp);
        std::fread(&revision, sizeof(int), 1, fp);
        if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
            size_t seen;
            std::fread(&seen, sizeof(size_t), 1, fp);
        } else {
            int iseen = 0;
            std::fread(&iseen, sizeof(int), 1, fp);
        }
        // int transpose = (major > 1000) || (minor > 1000);

        float val;
        for(unsigned int i=0;i!=weight_size.size();++i){
            int cap = 4*channels[i]+weight_size[i];
            if (i+1==weight_size.size()) cap = weight_size[i]+channels[i];
            for (int j=0;j!=cap;++j) {
                std::fread(&val, sizeof(float), 1, fp);
                weight_out[i].write(val);
            }
        }
        fprintf(stderr, "Done!\n");
        fclose(fp);
    }
};

class InputLayer: kahn_process{
public:
    const std::string filename;
    const int w;
    const int h;

    sc_fifo_out<float> act_out;
    sc_fifo_in<bool> ctrl_in;
    sc_fifo_out<bool> ctrl_out;

    InputLayer(sc_module_name name, const char* fname, int _w, int _h):
        kahn_process(name), filename(fname), w(_w), h(_h) {}
    
    void process() override{
        bool ctrl;
        ctrl_in.read(ctrl);
        ctrl_out.write(ctrl);

        image im = load_image_color(filename.c_str(),0,0);
        image sized = letterbox_image(im, w, h);

        for (int i=0;i!=h;++i)
            for (int j=0;j!=w;++j)
                for (int k=0;k!=3;++k)
                {
                    act_out.write(sized.data[j+w*(i+k*h)]);
                }

        free_image(im);
        free_image(sized);
    }
};

class kpn_yolo: public sc_module{
public:
    InputLayer *in;
    ConvLayer *l0;
    MaxPoolingLayer *l1;
    ConvLayer *l2;
    MaxPoolingLayer *l3;
    ConvLayer *l4;
    MaxPoolingLayer *l5;
    ConvLayer *l6;
    MaxPoolingLayer *l7;
    ConvLayer *l8;
    MaxPoolingLayer *l9;
    ConvLayer *l10;
    MaxPoolingLayer *l11;
    ConvLayer *l12;
    ConvLayer *l13;
    ConvLayer *l14;
    RegionLayer *l15;

    WeightLoader *loader;

    sc_fifo<float> **acts, **weights;
    sc_fifo<bool> **ctrls;

    sc_fifo_out<bool> ctrl_out;

    SC_HAS_PROCESS(kpn_yolo);

    kpn_yolo(sc_module_name name, const char *fname):sc_module(name) {
        in = new InputLayer("Input", fname, 416, 416);
        l0 = new ConvLayer("Conv0", 416, 416, 16, 3, 1, 3, 1, true, true);
        l1 = new MaxPoolingLayer("Max1", 416, 416, 16, 2, 2);
        l2 = new ConvLayer("Conv2", 208, 208, 32, 16, 1, 3, 1, true, true);
        l3 = new MaxPoolingLayer("Max3", 208, 208, 32, 2, 2);
        l4 = new ConvLayer("Conv4", 104, 104, 64, 32, 1, 3, 1, true, true);
        l5 = new MaxPoolingLayer("Max5", 104, 104, 64, 2, 2);
        l6 = new ConvLayer("Conv6", 52, 52, 128, 64, 1, 3, 1, true, true);
        l7 = new MaxPoolingLayer("Max7", 52, 52, 128, 2, 2);
        l8 = new ConvLayer("Conv8", 26, 26, 256, 128, 1, 3, 1, true, true);
        l9 = new MaxPoolingLayer("Max9", 26, 26, 256, 2, 2);
        l10 = new ConvLayer("Conv10", 13, 13, 512, 256, 1, 3, 1, true, true);
        l11 = new MaxPoolingLayer("Max11", 13, 13, 512, 1, 2);
        l12 = new ConvLayer("Conv12", 13, 13, 1024, 512, 1, 3, 1, true, true);
        l13 = new ConvLayer("Conv13", 13, 13, 512, 1024, 1, 3, 1, true, true);
        l14 = new ConvLayer("Conv14", 13, 13, 425, 512, 1, 1, 0, false, false);
        l15 = new RegionLayer("Region15", fname, 13, 13, 425, 416, 416);

        loader = new WeightLoader("WeightLoader", "yolov2-tiny.cfg","yolov2-tiny.weights", 9);

        acts = new sc_fifo<float>*[16];
        for (int i=0;i!=16;++i) acts[i] = new sc_fifo<float>;
        in->act_out(*acts[0]);
        l0->act_out(*acts[1]);
        l1->act_out(*acts[2]);
        l2->act_out(*acts[3]);
        l3->act_out(*acts[4]);
        l4->act_out(*acts[5]);
        l5->act_out(*acts[6]);
        l6->act_out(*acts[7]);
        l7->act_out(*acts[8]);
        l8->act_out(*acts[9]);
        l9->act_out(*acts[10]);
        l10->act_out(*acts[11]);
        l11->act_out(*acts[12]);
        l12->act_out(*acts[13]);
        l13->act_out(*acts[14]);
        l14->act_out(*acts[15]);

        l0->act_in(*acts[0]);
        l1->act_in(*acts[1]);
        l2->act_in(*acts[2]);
        l3->act_in(*acts[3]);
        l4->act_in(*acts[4]);
        l5->act_in(*acts[5]);
        l6->act_in(*acts[6]);
        l7->act_in(*acts[7]);
        l8->act_in(*acts[8]);
        l9->act_in(*acts[9]);
        l10->act_in(*acts[10]);
        l11->act_in(*acts[11]);
        l12->act_in(*acts[12]);
        l13->act_in(*acts[13]);
        l14->act_in(*acts[14]);
        l15->act_in(*acts[15]);

        weights = new sc_fifo<float>*[9];
        for (int i=0;i!=9;++i) weights[i] = new sc_fifo<float>;
        for (int i=0;i!=9;++i) loader->weight_out[i](*weights[i]);
        l0->weight_in(*weights[0]);
        l2->weight_in(*weights[1]);
        l4->weight_in(*weights[2]);
        l6->weight_in(*weights[3]);
        l8->weight_in(*weights[4]);
        l10->weight_in(*weights[5]);
        l12->weight_in(*weights[6]);
        l13->weight_in(*weights[7]);
        l14->weight_in(*weights[8]);

        ctrls = new sc_fifo<bool>*[18];
        for (int i=0;i!=18;++i) ctrls[i] = new sc_fifo<bool>;
        ctrl_out(*ctrls[17]);
        in->ctrl_out(*ctrls[0]);
        loader->ctrl_out(*ctrls[1]);
        l0->ctrl_out(*ctrls[2]);
        l1->ctrl_out(*ctrls[3]);
        l2->ctrl_out(*ctrls[4]);
        l3->ctrl_out(*ctrls[5]);
        l4->ctrl_out(*ctrls[6]);
        l5->ctrl_out(*ctrls[7]);
        l6->ctrl_out(*ctrls[8]);
        l7->ctrl_out(*ctrls[9]);
        l8->ctrl_out(*ctrls[10]);
        l9->ctrl_out(*ctrls[11]);
        l10->ctrl_out(*ctrls[12]);
        l11->ctrl_out(*ctrls[13]);
        l12->ctrl_out(*ctrls[14]);
        l13->ctrl_out(*ctrls[15]);
        l14->ctrl_out(*ctrls[16]);

        in->ctrl_in(*ctrls[17]);
        loader->ctrl_in(*ctrls[0]);
        l0->ctrl_in(*ctrls[1]);
        l1->ctrl_in(*ctrls[2]);
        l2->ctrl_in(*ctrls[3]);
        l3->ctrl_in(*ctrls[4]);
        l4->ctrl_in(*ctrls[5]);
        l5->ctrl_in(*ctrls[6]);
        l6->ctrl_in(*ctrls[7]);
        l7->ctrl_in(*ctrls[8]);
        l8->ctrl_in(*ctrls[9]);
        l9->ctrl_in(*ctrls[10]);
        l10->ctrl_in(*ctrls[11]);
        l11->ctrl_in(*ctrls[12]);
        l12->ctrl_in(*ctrls[13]);
        l13->ctrl_in(*ctrls[14]);
        l14->ctrl_in(*ctrls[15]);
        l15->ctrl_in(*ctrls[16]);

        SC_THREAD(start);
    }

    ~kpn_yolo() {
        for (int i=0;i!=18;++i) delete ctrls[i];
        delete ctrls;

        for (int i=0;i!=9;++i) delete weights[i];
        delete weights;

        for (int i=0;i!=16;++i) delete acts[i];
        delete acts;

        delete loader;
        delete l14;
        delete l13;
        delete l12;
        delete l11;
        delete l10;
        delete l9;
        delete l8;
        delete l7;
        delete l6;
        delete l5;
        delete l4;
        delete l3;
        delete l2;
        delete l1;
        delete l0;
        delete in;
    }

    void start() {
        ctrl_out.write(1);
    }
};

int	sc_main(int argc, char *argv[]) 
{
	kpn_yolo knn0("kpn_yolo", argv[1]);
	sc_start();
	return 0;
}


