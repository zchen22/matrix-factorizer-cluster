#include "gpu.h"

Gpu::Gpu(const int id, MpiInfo* mpi_info, Logger* logger)
    : id_(id), mpi_info_(mpi_info), logger_(logger) {
  cudaError_t e = cudaSuccess;
  e = cudaStreamCreate(&shader_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&h2d_stream_);
  logger_->CheckCudaError(e);
  e = cudaStreamCreate(&d2h_stream_);
  logger_->CheckCudaError(e);
  e = cudaMemGetInfo(&free_global_mem_size_, &global_mem_size_);
  logger->CheckCudaError(e);
}

Gpu::~Gpu() {
}

