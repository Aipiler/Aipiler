#ifndef GLOBALS_H
#define GLOBALS_H
#include "memref.h"

// Layer rms.weight: shape torch.Size([5120]), dtype torch.float32
extern "C" { RankedMemRefType<half, 1> *rms_weight; }
void init_rms_weight(){  
    half *rms_weight_data = new half[5120];
    int64_t rms_weight_shape[1] = {5120};
    rms_weight =
        new RankedMemRefType<half, 1>(rms_weight_data, rms_weight_shape);
}

void delete_rms_weight(){  
    delete rms_weight;
}

void init_all_globals() {
	init_rms_weight();
std::vector<std::string> model_names = {"rms_model.bin"};
std::map<std::string, half*> param_and_loc =
{
{"rms.weight", rms_weight->data}
};
mix::utils::load_model_f16(model_names, param_and_loc);
}
void delete_all_globals() {
	delete_rms_weight();
}
#endif
