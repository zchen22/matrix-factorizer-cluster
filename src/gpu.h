#ifndef GPU_H_
#define GPU_H_

// CUDA header
#include <cuda_runtime.h>

// Project headers
#include "logger.h"
#include "mpi-info.h"

class Gpu {
 public:
  Gpu(const int id, MpiInfo* mpi_info, Logger* logger);
  ~Gpu();
  // Getters
  cudaStream_t GetShaderStream() const { return shader_stream_; }
  cudaStream_t GetH2dStream() const { return h2d_stream_; }
  cudaStream_t GetD2hStream() const { return d2h_stream_; }
  size_t GetGlobalMemSize() const { return global_mem_size_; }
 private:
  int id_;
  cudaStream_t shader_stream_;
  cudaStream_t h2d_stream_;
  cudaStream_t d2h_stream_;
  size_t global_mem_size_;
  size_t free_global_mem_size_;
  // MPI info
  MpiInfo* mpi_info_;
  // Logger
  Logger* logger_;
};

#endif

