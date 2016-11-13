#ifndef FEATURE_MATRIX_H_
#define FEATURE_MATRIX_H_

// C++ headers
#include <string>
#include <vector>

// Project headers
#include "configuration-set.h"
#include "logger.h"
#include "mpi-info.h"

class FeatureMatrix {
 public:
  FeatureMatrix(const std::string name, const int num_rows,
                ConfigurationSet* config, MpiInfo* mpi_info, Logger* logger);
  ~FeatureMatrix();
  // Getters
  float GetFeature(const int64_t index) const {
    return feature_1d_vector_[index];
  }
  float* GetFeatureDev() const { return feature_1d_vector_dev_; }
  float* GetGradientDev() const { return gradient_1d_vector_dev_; }
  // Initialize features
  int Initialize();
  // Flatten
  int Flatten();
  // CPU/GPU memory transfer
  int AllocateGpuMemory();
  int CopyToGpu();
  int CopyToCpuFlatten();
  // Inter-node data transfer
  int CopyFeaturesToNextNode();
  int CompressFeaturesToNextNode(
      const std::vector<bool>& has_rating_vector,
      const std::vector<bool>& next_node_has_rating_vector);
  int CompressGradientsToNextNode(
      const std::vector<bool>& has_rating_vector,
      const std::vector<bool>& next_node_has_rating_vector);
  int LaunchCopyFeaturesToNextNode();
  int LaunchCopyGradientsToNextNode();
  int WaitCopyToNextNode();
  int DecompressFeatures();
  int DecompressGradients();
  // Dump to file
  int Dump1dToFile();
 private:
  // Name
  std::string name_;
  // Data
  std::vector<std::vector<float>> feature_vector_;
  std::vector<std::vector<float>> gradient_vector_;
  // 1D data
  std::vector<float> feature_1d_vector_;
  std::vector<float> gradient_1d_vector_;
  // Metadata
  int num_rows_;
  ConfigurationSet* config_;
  // GPU memory objects
  float* feature_1d_vector_dev_;
  float* gradient_1d_vector_dev_;
  // MPI data
  MpiInfo* mpi_info_;
  std::vector<float> send_feature_1d_vector_;
  std::vector<float> recv_feature_1d_vector_;
  std::vector<float> send_gradient_1d_vector_;
  std::vector<float> recv_gradient_1d_vector_;
  std::vector<MPI_Request> copy_req_vector_;
  // File that stores feature values
  std::string filename_;
  FILE* file_;
  // Logger
  Logger* logger_;
};

#endif

