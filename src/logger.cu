#include "logger.h"

// C++ headers
#include <cassert>
#include <ctime>

// MPI header
#include <mpi.h>

Logger::Logger(MpiInfo* mpi_info)
    : mpi_info_(mpi_info) {
  // Generate filename
  time_t timer = time(NULL);
  tm local_time = *localtime(&timer);
  char time_string[1024] = {0};
  snprintf(time_string, sizeof time_string,
           "%04d-%02d-%02d-%02d-%02d-%02d.log", mpi_info_->comm_rank,
           local_time.tm_mon, local_time.tm_mday, local_time.tm_hour,
           local_time.tm_min, local_time.tm_sec);
  filename_ = std::string("mf-") + time_string;
  // Open file
  file_ = fopen(filename_.c_str(), "w");
  if (!file_) {
    Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
}

Logger::~Logger() {
  // Close file
  fclose(file_);
}

int Logger::Warning(const char* format, ...) {
  fprintf(file_, "%-8s (%2d): ", "WARNING", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Info(const char* format, ...) {
  fprintf(file_, "%-8s (%2d): ", "INFO", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Debug(const char* format, ...) {
  fprintf(file_, "%-8s (%2d): ", "DEBUG", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file_, format, args);
  va_end(args);
  fflush(file_);
  return 0;
}

int Logger::Error(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s (%2d): ", "ERROR", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  MPI_Finalize();
  exit(EXIT_FAILURE);
}

int Logger::Warning(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s (%2d): ", "WARNING", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::Info(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s (%2d): ", "INFO", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::Debug(FILE* file, const char* format, ...) {
  fprintf(file, "%-8s (%2d): ", "DEBUG", mpi_info_->comm_rank);
  va_list args;
  va_start(args, format);
  vfprintf(file, format, args);
  va_end(args);
  fflush(file);
  return 0;
}

int Logger::CheckCudaError(const cudaError_t e) {
  if (e != cudaSuccess) {
    Error(stderr, "%s\n", cudaGetErrorString(e));
  }
  return 0;
}

int Logger::CheckMpiError(const int e) {
  if (e == MPI_SUCCESS) {
    return 0;
  }
  char error_string[MPI_MAX_ERROR_STRING];
  int error_string_len = 0;
  int ret = MPI_SUCCESS;
  ret = MPI_Error_string(e, error_string, &error_string_len);
  assert(ret == MPI_SUCCESS);
  Error(stderr, "%s\n", error_string);
  return 0;
}

int Logger::StartTimer() {
  gettimeofday(&start_time_, NULL);
  return 0;
}

int Logger::StopTimer() {
  gettimeofday(&end_time_, NULL);
  return 0;
}

