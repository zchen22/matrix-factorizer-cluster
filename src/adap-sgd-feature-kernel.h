#ifndef ADAP_SGD_FEATURE_KERNEL_H_
#define ADAP_SGD_FEATURE_KERNEL_H_

// C++ header
#include <cstdint>

__global__ void AdapSgdFeature(const int64_t record_id_base,
                               const int* __restrict__ user_ids,
                               const int* __restrict__ item_ids,
                               const int user_id_base,
                               const int item_id_base,
                               const float* __restrict__ ratings,
                               const int num_features,
                               const float learning_rate,
                               const float regularization_factor,
                               float* __restrict__ user_grads,
                               float* __restrict__ item_grads,
                               float* __restrict__ user_features,
                               float* __restrict__ item_features);

#endif

