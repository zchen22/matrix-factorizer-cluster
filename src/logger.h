#ifndef LOGGER_H_
#define LOGGER_H_

// C++ headers
#include <cstdarg>
#include <string>

// Linux header
#include <sys/time.h>

// CUDA header
#include <cuda_runtime.h>

// Project header
#include "mpi-info.h"

class Logger {
 public:
  Logger(MpiInfo* mpi_info);
  ~Logger();
  // Logging
  int Warning(const char* format, ...);
  int Info(const char* format, ...);
  int Debug(const char* format, ...);
  int Error(FILE* file, const char* format, ...);
  int Warning(FILE* file, const char* format, ...);
  int Info(FILE* file, const char* format, ...);
  int Debug(FILE* file, const char* format, ...);
  // Error checking
  int CheckCudaError(const cudaError_t e);
  int CheckMpiError(const int e);
  // Timing
  int StartTimer();
  int StopTimer();
  double ReadTimer() const {
    return (1. * (end_time_.tv_sec * 1e6 + end_time_.tv_usec) -
      (start_time_.tv_sec * 1e6 + start_time_.tv_usec)) / 1e6;
  }
 private:
  // Log file
  std::string filename_;
  FILE* file_;
  // Timers
  timeval start_time_;
  timeval end_time_;
  // MPI info
  MpiInfo* mpi_info_;
};

#endif

