#include "matrix-factorizer.h"

// C++ headers
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <sstream>

// Linux headers
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// Project headers
#include "adap-sgd-feature-kernel.h"
#include "sgd-feature-kernel.h"

MatrixFactorizer::MatrixFactorizer(MpiInfo* mpi_info)
    : train_dataset_(NULL), test_dataset_(NULL), user_features_(NULL),
      item_features_(NULL), train_rmse_(0), test_rmse_(0), gpu_(NULL),
      mpi_info_(mpi_info) {
  logger_ = new Logger(mpi_info_);
  config_ = new ConfigurationSet(mpi_info_, logger_);
  srand(1);
}

MatrixFactorizer::~MatrixFactorizer() {
  delete gpu_;
  delete test_dataset_;
  delete train_dataset_;
  delete config_;
  delete logger_;
}

int MatrixFactorizer::Setup(
    std::unordered_map<std::string, std::string>& arg_map) {
  assert(arg_map.find("-t") != arg_map.end());
  train_dataset_ = new Dataset("train", arg_map["-t"], mpi_info_, logger_);
  train_dataset_->Load();
  logger_->Debug(stderr, "Train dataset loaded\n");
  if (arg_map.find("-e") != arg_map.end()) {
    test_dataset_ = new Dataset("test", arg_map["-e"], mpi_info_, logger_);
    test_dataset_->Load();
    logger_->Debug(stderr, "Test dataset loaded\n");
  }
  if (arg_map.find("-c") != arg_map.end()) {
    config_->Load(arg_map["-c"]);
    logger_->Debug(stderr, "%s\n", config_->ToString().c_str());
  }
  gpu_ = new Gpu(0, mpi_info_, logger_);
  float a = gpu_->GetGlobalMemSize();
  float b = -4.f * config_->num_features * (train_dataset_->GetNumUsers() +
      train_dataset_->GetNumItems());
  float c = -12.f * train_dataset_->GetNumRecords();
  logger_->Debug(stderr, "a = %f b = %f c = %f\n", a, b, c);
  logger_->Debug(stderr, "Min number of nodes required = %d\n",
                 ceil((-b + sqrt(b * b - 4 * a * c)) / (2 * a)));
  return 0;
}

int MatrixFactorizer::InitializeFeatures() {
  logger_->Info(stderr, "Initializing features...\n");
  train_dataset_->ComputeBlockSizes();
  if (test_dataset_) {
    test_dataset_->ComputeBlockSizes();
  }
  user_features_ = new FeatureMatrix("user", train_dataset_->GetUserBlockSize(),
                                     config_, mpi_info_, logger_);
  user_features_->Initialize();
  item_features_ = new FeatureMatrix("item", train_dataset_->GetItemBlockSize(),
                                     config_, mpi_info_, logger_);
  item_features_->Initialize();
  logger_->Info(stderr, "Features initialized\n");
  return 0;
}

int MatrixFactorizer::Preprocess() {
  logger_->Info(stderr, "Preprocessing data\n");
  if (config_->decomp_mode == ConfigurationSet::kDecompModeRecord) {
    assert(false);
  } else if (config_->decomp_mode == ConfigurationSet::kDecompModeRow) {
    assert(false);
  } else if (config_->decomp_mode == ConfigurationSet::kDecompModeFeature) {
    ShuffleUserIds();
    ShuffleItemIds();
  } else {
    assert(false);
  }
  train_dataset_->GenerateLocalBlocks();
  train_dataset_->ShuffleLocalBlocks();
  train_dataset_->FlattenLocalBlocks();
  train_dataset_->GenerateLocalBlocksCoo();
  train_dataset_->FlattenLocalBlocksCoo();
  if (test_dataset_) {
    test_dataset_->GenerateLocalBlocks();
    test_dataset_->FlattenLocalBlocks();
    test_dataset_->GenerateLocalBlocksCoo();
    test_dataset_->FlattenLocalBlocksCoo();
  }
  user_features_->Flatten();
  item_features_->Flatten();
  LabelUntrainedItems();
  logger_->Info(stderr, "Data preprocessed\n");
  return 0;
}

int MatrixFactorizer::ShuffleUserIds() {
  std::vector<int> new_id_vector;
  for (int i = 0; i < train_dataset_->GetNumUsers(); ++i) {
    new_id_vector.push_back(i);
  }
  std::random_shuffle(new_id_vector.begin(), new_id_vector.end());
  logger_->Debug(stderr, "user 0's new id = %d\n", new_id_vector[0]);
  train_dataset_->AssignNewUserIds(new_id_vector);
  if (test_dataset_) {
    test_dataset_->AssignNewUserIds(new_id_vector);
  }
  return 0;
}

int MatrixFactorizer::ShuffleItemIds() {
  std::vector<int> new_id_vector;
  for (int i = 0; i < train_dataset_->GetNumItems(); ++i) {
    new_id_vector.push_back(i);
  }
  std::random_shuffle(new_id_vector.begin(), new_id_vector.end());
  logger_->Debug(stderr, "item 0's new id = %d\n", new_id_vector[0]);
  train_dataset_->AssignNewItemIds(new_id_vector);
  if (test_dataset_) {
    test_dataset_->AssignNewItemIds(new_id_vector);
  }
  return 0;
}

int MatrixFactorizer::LabelUntrainedItems() {
  has_rating_vector_.assign(mpi_info_->comm_size, std::vector<bool>());
  for (auto& b : has_rating_vector_) {
    b.assign(train_dataset_->GetItemBlockSize(), false);
  }
  for (const auto& r : train_dataset_->GetRecords()) {
    const int item_block_id = r.item_id / train_dataset_->GetItemBlockSize();
    const int item_local_id = r.item_id % train_dataset_->GetItemBlockSize();
    has_rating_vector_[item_block_id][item_local_id] = true;
  }
  int num_untrained_items = 0;
  for (const auto& b : has_rating_vector_) {
    for (const auto& i : b) {
      if (!i) {
        ++num_untrained_items;
      }
    }
  }
  logger_->Debug("%d items not be trained\n", num_untrained_items);
  return 0;
}

int MatrixFactorizer::AllocateGpuMemory() {
  logger_->Info(stderr, "Allocating GPU memory...\n");
  train_dataset_->AllocateGpuMemory();
  user_features_->AllocateGpuMemory();
  item_features_->AllocateGpuMemory();
  logger_->Info(stderr, "GPU memory allocated\n");
  return 0;
}

int MatrixFactorizer::CopyToGpu() {
  logger_->Info(stderr, "Copying data to GPU...\n");
  train_dataset_->CopyToGpu();
  user_features_->CopyToGpu();
  item_features_->CopyToGpu();
  logger_->Info(stderr, "Data copied\n");
  return 0;
}

int MatrixFactorizer::Train() {
  if (config_->show_train_rmse) {
    ComputeTrainRmse();
  }
  if (config_->show_test_rmse && test_dataset_) {
    ComputeTestRmse();
  }
  for (int iter = 1; iter <= config_->max_num_iterations; ++iter) {
    logger_->Info(stderr, "Starting iteration %-4d\n", iter);
    for (int block_id_offset = 0; block_id_offset < mpi_info_->comm_size;
         ++block_id_offset) {
      const int item_block_id = (mpi_info_->comm_rank + block_id_offset) %
          mpi_info_->comm_size;
      switch (config_->gd_mode) {
      case ConfigurationSet::kGdModeMiniBatchSgd:
        if (config_->batch_size == 1) {
          LaunchSgdFeatureKernel(item_block_id);
        } else {
          assert(false);
        }
        break;
      case ConfigurationSet::kGdModeAdapSgd:
        LaunchAdapSgdFeatureKernel(item_block_id);
        break;
      default: assert(false);
      }
      cudaError_t e = cudaSuccess;
      e = cudaStreamSynchronize(gpu_->GetShaderStream());
      logger_->CheckCudaError(e);
      logger_->StopTimer();
      logger_->Info("Kernel time = %-8g\n", logger_->ReadTimer());
      // Prepare data for next iteration
      if (mpi_info_->comm_size > 1) {
        logger_->StartTimer();
        item_features_->CopyToCpuFlatten();
        logger_->StopTimer();
        logger_->Info("G2C time = %-8g\n", logger_->ReadTimer());
        int ret = MPI_Barrier(mpi_info_->comm);
        logger_->CheckMpiError(ret);
        logger_->StartTimer();
        item_features_->CompressFeaturesToNextNode(
            has_rating_vector_[item_block_id],
            has_rating_vector_[(item_block_id + 1) % mpi_info_->comm_size]);
        item_features_->LaunchCopyFeaturesToNextNode();
        if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
          item_features_->CompressGradientsToNextNode(
              has_rating_vector_[item_block_id],
              has_rating_vector_[(item_block_id + 1) % mpi_info_->comm_size]);
          item_features_->LaunchCopyGradientsToNextNode();
        }
        item_features_->WaitCopyToNextNode();
        item_features_->DecompressFeatures();
        if (config_->gd_mode == ConfigurationSet::kGdModeAdapSgd) {
          item_features_->DecompressGradients();
        }
        logger_->StopTimer();
        logger_->Info("MPI time = %-8g\n", logger_->ReadTimer());
        logger_->StartTimer();
        item_features_->CopyToGpu();
        logger_->StopTimer();
        logger_->Info("C2G time = %-8g\n", logger_->ReadTimer());
      }
    }
    if (config_->show_train_rmse ||
        (config_->show_test_rmse && test_dataset_)) {
      user_features_->CopyToCpuFlatten();
      item_features_->CopyToCpuFlatten();
    }
    if (config_->show_train_rmse) {
      ComputeTrainRmse();
    }
    if (config_->show_test_rmse && test_dataset_) {
      ComputeTestRmse();
    }
  }
  return 0;
}

int MatrixFactorizer::LaunchSgdFeatureKernel(const int item_block_id) {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetBlockNumRecords(item_block_id), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  const int64_t record_id_base = train_dataset_->GetBlockRecordIdBase(
      item_block_id);
  const int user_id_base = train_dataset_->GetUserBlockSize() *
      mpi_info_->comm_rank;
  const int item_id_base = train_dataset_->GetItemBlockSize() *
      item_block_id;
  logger_->Debug("Kernel size = (%d, %d)\n", grid_size.x, block_size.x);
  logger_->Info(stderr, "Working on item block %d...\n", item_block_id);
  logger_->StartTimer();
  SgdFeature<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      record_id_base, train_dataset_->GetUserIdBlockDev(),
      train_dataset_->GetItemIdBlockDev(), user_id_base, item_id_base,
      train_dataset_->GetRatingBlockDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  return 0;
}

int MatrixFactorizer::LaunchAdapSgdFeatureKernel(const int item_block_id) {
  cudaError_t e = cudaSuccess;
  e = cudaStreamSynchronize(gpu_->GetH2dStream());
  logger_->CheckCudaError(e);
  dim3 grid_size(train_dataset_->GetBlockNumRecords(item_block_id), 1, 1);
  dim3 block_size(config_->num_features, 1, 1);
  const int64_t record_id_base = train_dataset_->GetBlockRecordIdBase(
      item_block_id);
  const int user_id_base = train_dataset_->GetUserBlockSize() *
      mpi_info_->comm_rank;
  const int item_id_base = train_dataset_->GetItemBlockSize() *
      item_block_id;
  logger_->Debug("Kernel size = (%d, %d)\n", grid_size.x, block_size.x);
  logger_->Info(stderr, "Working on item block %d...\n", item_block_id);
  logger_->StartTimer();
  AdapSgdFeature<<<grid_size, block_size,
      config_->num_features * sizeof(float), gpu_->GetShaderStream()>>>(
      record_id_base, train_dataset_->GetUserIdBlockDev(),
      train_dataset_->GetItemIdBlockDev(), user_id_base, item_id_base,
      train_dataset_->GetRatingBlockDev(), config_->num_features,
      config_->learning_rate, config_->regularization_factor,
      user_features_->GetGradientDev(), item_features_->GetGradientDev(),
      user_features_->GetFeatureDev(), item_features_->GetFeatureDev());
  return 0;
}

int MatrixFactorizer::Sync() {
  int ret = MPI_SUCCESS;
  ret = MPI_Barrier(mpi_info_->comm);
  logger_->CheckMpiError(ret);
  return 0;
}

int MatrixFactorizer::ComputeTrainRmse() {
  logger_->Info(stderr, "Computing training rmse...\n");
  float error = 0;
  error = ComputeSquareErrorSum(train_dataset_);
  train_rmse_ = sqrt(error / train_dataset_->GetNumRecords());
  logger_->Info(stderr, "Training rmse = %f\n", train_rmse_);
  return 0;
}

int MatrixFactorizer::ComputeTestRmse() {
  logger_->Info(stderr, "Computing testing rmse...\n");
  float error = 0;
  error = ComputeSquareErrorSum(test_dataset_);
  test_rmse_ = sqrt(error / test_dataset_->GetNumRecords());
  logger_->Info(stderr, "Testing rmse = %f\n", test_rmse_);
  return 0;
}

int MatrixFactorizer::DumpFeatures() {
  user_features_->CopyToCpuFlatten();
  item_features_->CopyToCpuFlatten();
  user_features_->Dump1dToFile();
  item_features_->Dump1dToFile();
  return 0;
}

float MatrixFactorizer::ComputeSquareErrorSum(Dataset* dataset) {
  float error = 0;
  float c = 0;
  const int user_id_base = dataset->GetUserBlockSize() *
      mpi_info_->comm_rank;
  for (int block_id_offset = 0; block_id_offset < mpi_info_->comm_size;
       ++block_id_offset) {
    const int item_block_id = (mpi_info_->comm_rank + block_id_offset) %
        mpi_info_->comm_size;
    const int item_id_base = dataset->GetItemBlockSize() * item_block_id;
    for (int record_id_offset = 0;
         record_id_offset < dataset->GetBlockNumRecords(item_block_id);
         ++record_id_offset) {
      const int64_t record_id = dataset->GetBlockRecordIdBase(
          item_block_id) + record_id_offset;
      const int user_id_local = dataset->GetUserIdBlock(record_id) -
          user_id_base;
      const int item_id_local = dataset->GetItemIdBlock(record_id) -
          item_id_base;
      float e = 0;
      e = PredictRating(user_id_local, item_id_local) -
          dataset->GetRatingBlock(record_id);
      e *= e;
      // Kahan sum algorithm
      float y = e - c;
      float t = error + y;
      c = (t - error) - y;
      error = t;
    }
    if (mpi_info_->comm_size > 1) {
      item_features_->CopyFeaturesToNextNode();
    }
  }
  float error_sum = 0;
  int ret = MPI_SUCCESS;
  ret = MPI_Allreduce(&error, &error_sum, 1, MPI_FLOAT, MPI_SUM,
                      mpi_info_->comm);
  logger_->CheckMpiError(ret);
  return error_sum;
}

float MatrixFactorizer::PredictRating(const int64_t user_id_local,
                                      const int64_t item_id_local) {
  float rating = 0;
  for (int feature_id = 0; feature_id < config_->num_features; ++feature_id) {
    float user_feature = user_features_->GetFeature(
        user_id_local * config_->num_features + feature_id);
    float item_feature = item_features_->GetFeature(
        item_id_local * config_->num_features + feature_id);
    rating += user_feature * item_feature;
  }
  return rating;
}

