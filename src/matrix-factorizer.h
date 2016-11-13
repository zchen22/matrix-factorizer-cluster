#ifndef MATRIX_FACTORIZER_H_
#define MATRIX_FACTORIZER_H_

// C++ header
#include <unordered_map>

// Project headers
#include "configuration-set.h"
#include "dataset.h"
#include "feature-matrix.h"
#include "gpu.h"
#include "logger.h"
#include "mpi-info.h"
#include "record.h"

class MatrixFactorizer {
 public:
  MatrixFactorizer(MpiInfo* mpi_info);
  ~MatrixFactorizer();
  // Setup datasets, configurations, etc.
  int Setup(std::unordered_map<std::string, std::string>& arg_map);
  // Initialization
  int InitializeFeatures();
  // Preprocessing
  int Preprocess();
  int ShuffleUserIds();
  int ShuffleItemIds();
  int LabelUntrainedItems();
  // GPU memory
  int AllocateGpuMemory();
  int CopyToGpu();
  // Training
  int Train();
  int LaunchSgdFeatureKernel(const int item_block_id);
  int LaunchAdapSgdFeatureKernel(const int item_block_id);
  int Sync();
  // Statistics
  int ComputeTrainRmse();
  int ComputeTestRmse();
  // Output
  int DumpFeatures();
  // Helpers
  float ComputeSquareErrorSum(Dataset* dataset);
  float PredictRating(const int64_t user_id_local,
                      const int64_t item_id_local);
 private:
  // Datasets
  Dataset* train_dataset_;
  Dataset* test_dataset_;
  // Features
  FeatureMatrix* user_features_;
  FeatureMatrix* item_features_;
  // Configurations
  ConfigurationSet* config_;
  // Statistics
  float train_rmse_;
  float test_rmse_;
  std::vector<std::vector<bool>> has_rating_vector_;
  // GPU
  Gpu* gpu_;
  // MPI info
  MpiInfo* mpi_info_;
  // Logger
  Logger* logger_;
};

#endif

