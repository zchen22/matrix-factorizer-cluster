#include "feature-matrix.h"

// C++ headers
#include <cassert>
#include <cinttypes>
#include <ctime>
#include <fstream>
#include <random>

FeatureMatrix::FeatureMatrix(const std::string name, const int num_rows,
                             ConfigurationSet* config, MpiInfo* mpi_info,
                             Logger* logger)
    : num_rows_(num_rows), config_(config), feature_1d_vector_dev_(NULL),
      mpi_info_(mpi_info), file_(NULL), logger_(logger) {
  name_ = name;
  // Allocate memory for feature vector
  feature_vector_.assign(num_rows_, std::vector<float>());
  gradient_vector_.assign(num_rows_, std::vector<float>());
  for (int i = 0; i < num_rows_; ++i) {
    feature_vector_[i].assign(config_->num_features, 0);
    gradient_vector_[i].assign(config_->num_features, 0);
  }
  // Generate filename
  time_t timer = time(NULL);
  tm local_time = *localtime(&timer);
  char time_string[1024] = {0};
  snprintf(time_string, sizeof time_string,
           "%04d-%02d-%02d-%02d-%02d-%02d", mpi_info_->comm_rank,
           local_time.tm_mon, local_time.tm_mday, local_time.tm_hour,
           local_time.tm_min, local_time.tm_sec);
  filename_ = name_ + "-" + time_string + ".bin";
}

FeatureMatrix::~FeatureMatrix() {
}

int FeatureMatrix::Initialize() {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (int i = 0; i < num_rows_; ++i) {
    for (int j = 0; j < config_->num_features; ++j) {
      feature_vector_[i][j] = dist(rng);
      gradient_vector_[i][j] = 0;
    }
  }
  return 0;
}

int FeatureMatrix::Flatten() {
  feature_1d_vector_.reserve(num_rows_ * config_->num_features);
  gradient_1d_vector_.reserve(num_rows_ * config_->num_features);
  for (int i = 0; i < num_rows_; ++i) {
    for (int j = 0; j < config_->num_features; ++j) {
      feature_1d_vector_.push_back(feature_vector_[i][j]);
      gradient_1d_vector_.push_back(gradient_vector_[i][j]);
    }
  }
  return 0;
}

int FeatureMatrix::AllocateGpuMemory() {
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&feature_1d_vector_dev_,
                 feature_1d_vector_.size() * sizeof feature_1d_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&gradient_1d_vector_dev_,
                 gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0]);
  logger_->CheckCudaError(e);
  return 0;
}

int FeatureMatrix::CopyToGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMemcpy(feature_1d_vector_dev_, feature_1d_vector_.data(),
                 feature_1d_vector_.size() * sizeof feature_1d_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(gradient_1d_vector_dev_, gradient_1d_vector_.data(),
                 gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  return 0;
}

int FeatureMatrix::CopyToCpuFlatten() {
  cudaError_t e = cudaSuccess;
  e = cudaMemcpy(feature_1d_vector_.data(), feature_1d_vector_dev_,
                 feature_1d_vector_.size() * sizeof feature_1d_vector_[0],
                 cudaMemcpyDeviceToHost);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(gradient_1d_vector_.data(), gradient_1d_vector_dev_,
                 gradient_1d_vector_.size() * sizeof gradient_1d_vector_[0],
                 cudaMemcpyDeviceToHost);
  logger_->CheckCudaError(e);
  return 0;
}

int FeatureMatrix::CopyFeaturesToNextNode() {
  int ret = MPI_SUCCESS;
  int prev_node = mpi_info_->comm_rank != 0?
      mpi_info_->comm_rank - 1 : mpi_info_->comm_size - 1;
  MPI_Request reqs[2];
  ret = MPI_Isend(feature_1d_vector_.data(), feature_1d_vector_.size(),
                  MPI_FLOAT, prev_node, 0, mpi_info_->comm, &reqs[0]);
  logger_->CheckMpiError(ret);
  std::vector<float> new_feature_1d_vector(feature_1d_vector_.size(), 0);
  int next_node = mpi_info_->comm_rank != mpi_info_->comm_size - 1?
      mpi_info_->comm_rank + 1 : 0;
  ret = MPI_Irecv(new_feature_1d_vector.data(), feature_1d_vector_.size(),
                  MPI_FLOAT, next_node, 0, mpi_info_->comm, &reqs[1]);
  logger_->CheckMpiError(ret);
  ret = MPI_Waitall(sizeof reqs / sizeof(MPI_Request), reqs, MPI_STATUS_IGNORE);
  logger_->CheckMpiError(ret);
  feature_1d_vector_ = new_feature_1d_vector;
  return 0;
}

int FeatureMatrix::CompressFeaturesToNextNode(
    const std::vector<bool>& has_rating_vector,
    const std::vector<bool>& next_node_has_rating_vector) {
  int num_rows_transferred = 0;
  for (int64_t row = 0; row < num_rows_; ++row) {
    if (!has_rating_vector[row] && !next_node_has_rating_vector[row]) {
      continue;
    }
    send_feature_1d_vector_.push_back(row);
    for (int i = 0; i < config_->num_features; ++i) {
      send_feature_1d_vector_.push_back(
          feature_1d_vector_[row * config_->num_features + i]);
    }
    ++num_rows_transferred;
  }
  logger_->Info("Transferring %d out of %d feature vectors\n",
                num_rows_transferred, num_rows_);
  recv_feature_1d_vector_ = send_feature_1d_vector_;
  return 0;
}

int FeatureMatrix::CompressGradientsToNextNode(
    const std::vector<bool>& has_rating_vector,
    const std::vector<bool>& next_node_has_rating_vector) {
  int num_rows_transferred = 0;
  for (int64_t row = 0; row < num_rows_; ++row) {
    if (!has_rating_vector[row] && !next_node_has_rating_vector[row]) {
      continue;
    }
    send_gradient_1d_vector_.push_back(row);
    for (int i = 0; i < config_->num_features; ++i) {
      send_gradient_1d_vector_.push_back(
          gradient_1d_vector_[row * config_->num_features + i]);
    }
    ++num_rows_transferred;
  }
  logger_->Info("Transferring %d out of %d gradient vectors\n",
                num_rows_transferred, num_rows_);
  recv_gradient_1d_vector_ = send_gradient_1d_vector_;
  return 0;
}

int FeatureMatrix::LaunchCopyFeaturesToNextNode() {
  int ret = MPI_SUCCESS;
  int prev_node = mpi_info_->comm_rank != 0?
      mpi_info_->comm_rank - 1 : mpi_info_->comm_size - 1;
  int next_node = mpi_info_->comm_rank != mpi_info_->comm_size - 1?
      mpi_info_->comm_rank + 1 : 0;
  MPI_Request req;
  ret = MPI_Isend(send_feature_1d_vector_.data(),
                  send_feature_1d_vector_.size(), MPI_FLOAT, prev_node, 0,
                  mpi_info_->comm, &req);
  logger_->CheckMpiError(ret);
  logger_->Info("Transferring %zu features\n", send_feature_1d_vector_.size());
  copy_req_vector_.push_back(req);
  ret = MPI_Irecv(recv_feature_1d_vector_.data(),
                  recv_feature_1d_vector_.size(), MPI_FLOAT, next_node, 0,
                  mpi_info_->comm, &req);
  logger_->CheckMpiError(ret);
  copy_req_vector_.push_back(req);
  return 0;
}

int FeatureMatrix::LaunchCopyGradientsToNextNode() {
  int ret = MPI_SUCCESS;
  int prev_node = mpi_info_->comm_rank != 0?
      mpi_info_->comm_rank - 1 : mpi_info_->comm_size - 1;
  int next_node = mpi_info_->comm_rank != mpi_info_->comm_size - 1?
      mpi_info_->comm_rank + 1 : 0;
  MPI_Request req;
  ret = MPI_Isend(send_gradient_1d_vector_.data(),
                  send_gradient_1d_vector_.size(), MPI_FLOAT, prev_node, 0,
                  mpi_info_->comm, &req);
  logger_->CheckMpiError(ret);
  logger_->Info("Transferring %zu gradients\n",
                send_gradient_1d_vector_.size());
  copy_req_vector_.push_back(req);
  ret = MPI_Irecv(recv_gradient_1d_vector_.data(),
                  recv_gradient_1d_vector_.size(), MPI_FLOAT, next_node, 0,
                  mpi_info_->comm, &req);
  logger_->CheckMpiError(ret);
  copy_req_vector_.push_back(req);
  return 0;
}

int FeatureMatrix::WaitCopyToNextNode() {
  int ret = MPI_SUCCESS;
  ret = MPI_Waitall(copy_req_vector_.size(), copy_req_vector_.data(),
                    MPI_STATUS_IGNORE);
  logger_->CheckMpiError(ret);
  copy_req_vector_.clear();
  return 0;
}

int FeatureMatrix::DecompressFeatures() {
  for (int64_t i = 0; i < recv_feature_1d_vector_.size();
       i += 1 + config_->num_features) {
    const int64_t row = recv_feature_1d_vector_[i];
    for (int j = 0; j < config_->num_features; ++j) {
      feature_1d_vector_[row * config_->num_features + j] =
          recv_feature_1d_vector_[i + 1 + j];
    }
  }
  recv_feature_1d_vector_.clear();
  send_feature_1d_vector_.clear();
  return 0;
}

int FeatureMatrix::DecompressGradients() {
  for (int64_t i = 0; i < recv_gradient_1d_vector_.size();
       i += 1 + config_->num_features) {
    const int64_t row = recv_gradient_1d_vector_[i];
    for (int j = 0; j < config_->num_features; ++j) {
      gradient_1d_vector_[row * config_->num_features + j] =
          recv_gradient_1d_vector_[i + 1 + j];
    }
  }
  recv_gradient_1d_vector_.clear();
  send_gradient_1d_vector_.clear();
  return 0;
}

int FeatureMatrix::Dump1dToFile() {
  logger_->Info(stderr, "Dumping feature matrix '%s'...\n", name_.c_str());
  std::ofstream file(filename_, std::ofstream::binary);
  if (!file.is_open()) {
    logger_->Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
  union {
    char b[4];
    float f;
  } bytes = { .f = 0 };
  for (int64_t row = 0; row < num_rows_; ++row) {
    for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
      bytes.f = feature_1d_vector_[row * config_->num_features + feature_id];
      file.write(bytes.b, 4);
    }
  }
  file.close();
  logger_->Info(stderr, "Feature matrix '%s' dumped to file '%s'\n",
                name_.c_str(), filename_.c_str());
  return 0;
}

