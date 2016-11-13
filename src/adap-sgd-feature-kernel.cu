#include "adap-sgd-feature-kernel.h"

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
                               float* __restrict__ item_features) {
  // Compute error
  const int64_t record_id = record_id_base + blockIdx.x;
  const int64_t user_id_local = user_ids[record_id] - user_id_base;
  const int64_t item_id_local = item_ids[record_id] - item_id_base;
  const int64_t user_feature_idx = user_id_local * num_features + threadIdx.x;
  const int64_t item_feature_idx = item_id_local * num_features + threadIdx.x;
  const float user_feature = user_features[user_feature_idx];
  const float item_feature = item_features[item_feature_idx];
  extern __shared__ float partial_sums[];
  partial_sums[threadIdx.x] = user_feature * item_feature;
  __syncthreads();
  if (threadIdx.x < warpSize) {
    for (int i = threadIdx.x + warpSize; i < blockDim.x; i += warpSize) {
      partial_sums[threadIdx.x] += partial_sums[i];
    }
  }
  __syncthreads();
  float predicted_rating = partial_sums[threadIdx.x];
  if (threadIdx.x < warpSize) {
    for (int i = warpSize / 2; i >= 1; i /= 2) {
      predicted_rating += __shfl_xor(predicted_rating, i, warpSize);
    }
  }
  __syncthreads();
  __shared__ float error;
  if (threadIdx.x == 0) {
    error = predicted_rating - ratings[record_id];
  }
  __syncthreads();
  // Update features
  const float user_grad = regularization_factor * user_feature +
      error * item_feature;
  const float item_grad = regularization_factor * item_feature +
      error * user_feature;
  const float new_user_grad = user_grads[user_feature_idx] +
      user_grad * user_grad;
  const float new_item_grad = item_grads[item_feature_idx] +
      item_grad * item_grad;
  const float new_user_learning_rate = learning_rate / sqrt(new_user_grad);
  const float new_item_learning_rate = learning_rate / sqrt(new_item_grad);
  user_features[user_feature_idx] = user_feature -
      new_user_learning_rate * user_grad;
  item_features[item_feature_idx] = item_feature -
      new_item_learning_rate * item_grad;
  user_grads[user_feature_idx] = new_user_grad;
  item_grads[item_feature_idx] = new_item_grad;
}

