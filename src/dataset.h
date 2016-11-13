#ifndef DATASET_H_
#define DATASET_H_

// C++ headers
#include <cstdint>
#include <string>
#include <vector>

// Porject headers
#include "logger.h"
#include "mpi-info.h"
#include "record.h"

class Dataset {
 public:
  Dataset(const std::string name, const std::string filename,
          MpiInfo* mpi_info, Logger* logger);
  ~Dataset();
  // Getters
  const std::vector<Record>& GetRecords() const { return record_vector_; }
  int64_t GetNumRecords() const { return record_vector_.size(); }
  int GetNumUsers() const { return num_users_; }
  int GetNumItems() const { return num_items_; }
  int GetUserBlockSize() const { return user_block_size_; }
  int GetItemBlockSize() const { return item_block_size_; }
  int GetUserIdBlock(const int64_t record_id) const {
    return user_id_block_1d_vector_[record_id];
  }
  int GetItemIdBlock(const int64_t record_id) const {
    return item_id_block_1d_vector_[record_id];
  }
  float GetRatingBlock(const int64_t record_id) const {
    return rating_block_1d_vector_[record_id];
  }
  int64_t GetBlockRecordIdBase(const int index) const {
    return block_record_id_base_vector_[index];
  }
  int GetBlockNumRecords(const int index) const {
    return block_num_record_vector_[index];
  }
  int* GetUserIdBlockDev() const {
    return user_id_block_1d_vector_dev_;
  }
  int* GetItemIdBlockDev() const {
    return item_id_block_1d_vector_dev_;
  }
  float* GetRatingBlockDev() const {
    return rating_block_1d_vector_dev_;
  }
  // Load from dataset file
  int Load();
  int LoadFromText();
  int LoadFromBinary();
  // Shuffle
  int Shuffle();
  int AssignNewUserIds(const std::vector<int>& new_id_vector);
  int AssignNewItemIds(const std::vector<int>& new_id_vector);
  // Compute block sizes
  int ComputeBlockSizes();
  // Local blocks that will be processed on the local node
  int GenerateLocalBlocks();
  int ShuffleLocalBlocks();
  int GenerateLocalBlocksCoo();
  int FlattenLocalBlocks();
  int FlattenLocalBlocksCoo();
  // GPU memory
  int AllocateGpuMemory();
  int CopyToGpu();
  // Statistics
  int ComputeNumItemsEffective();
 private:
  // Name
  std::string name_;
  // Data
  std::vector<Record> record_vector_;
  int num_users_;
  int num_items_;
  // Data blocks that will be used by the local process
  std::vector<std::vector<Record>> record_block_vector_;
  std::vector<std::vector<int>> user_id_block_vector_;
  std::vector<std::vector<int>> item_id_block_vector_;
  std::vector<std::vector<float>> rating_block_vector_;
  int user_block_size_;
  int item_block_size_;
  // 1D data blocks that will be used by the local process
  std::vector<Record> record_block_1d_vector_;
  std::vector<int> user_id_block_1d_vector_;
  std::vector<int> item_id_block_1d_vector_;
  std::vector<float> rating_block_1d_vector_;
  std::vector<int64_t> block_record_id_base_vector_;
  std::vector<int> block_num_record_vector_;
  // GPU memory objects
  int* user_id_block_1d_vector_dev_;
  int* item_id_block_1d_vector_dev_;
  float* rating_block_1d_vector_dev_;
  // Statistics
  int num_items_effective_;
  // File that stores the data
  std::string filename_;
  // MPI info
  MpiInfo* mpi_info_;
  // Logger
  Logger* logger_;
};

#endif

