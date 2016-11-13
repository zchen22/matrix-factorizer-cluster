#include "dataset.h"

// C++ headers
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <fstream>
#include <sstream>

Dataset::Dataset(const std::string name, const std::string filename,
                 MpiInfo* mpi_info, Logger* logger)
    : num_users_(0), num_items_(0), user_block_size_(0), item_block_size_(0),
      user_id_block_1d_vector_dev_(NULL), item_id_block_1d_vector_dev_(NULL),
      rating_block_1d_vector_dev_(NULL), num_items_effective_(0),
      mpi_info_(mpi_info), logger_(logger) {
  name_ = name;
  filename_ = filename;
}

Dataset::~Dataset() {
}

int Dataset::Load() {
  if (filename_.compare(filename_.size() - 4, 4, ".txt") == 0) {
    LoadFromText();
  } else if (filename_.compare(filename_.size() - 4, 4, ".bin") == 0) {
    LoadFromBinary();
  } else {
    logger_->Error(stderr, "Unrecognized dataset file type\n");
  }
  return 0;
}

int Dataset::LoadFromText() {
  // Get the metadata
  std::ifstream file(filename_);
  if (!file.is_open()) {
    logger_->Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
  std::string line;
  int64_t num_records = 0;
  while (std::getline(file, line)) {
    if (line[0] == '%') {
      continue;
    }
    std::istringstream line_stream(line);
    if (num_users_ == 0) {
      assert(line_stream >> num_users_);
    } else if (num_users_ > 0) {
      int val = 0;
      assert(line_stream >> val);
      assert(val == num_users_);
    }
    if (num_items_ == 0) {
      assert(line_stream >> num_items_);
    } else if (num_items_ > 0) {
      int val = 0;
      assert(line_stream >> val);
      assert(val == num_items_);
    }
    assert(line_stream >> num_records);
    record_vector_.reserve(num_records);
    break;
  }
  // Get the data in the COO format
  for (int64_t i = 0; i < num_records; ++i) {
    assert(std::getline(file, line));
    int user_id = 0;
    int item_id = 0;
    float rating = 0;
    std::istringstream line_stream(line);
    assert(line_stream >> user_id);
    assert(line_stream >> item_id);
    assert(line_stream >> rating);
    record_vector_.push_back(Record(--user_id, --item_id, rating));
  }
  file.close();
  return 0;
}

int Dataset::LoadFromBinary() {
  // Get the metadata
  std::ifstream file(filename_, std::ifstream::binary);
  if (!file.is_open()) {
    logger_->Error(stderr, "File '%s' cannot be opened!\n", filename_.c_str());
  }
  union {
    char b[8];
    unsigned int u32;
    uint64_t u64;
    float f32;
    double f64;
  } bytes = { .u64 = 0 };
  file.read(bytes.b, 4);
  const unsigned int num1 = bytes.u32;
  bytes.u64 = 0;
  file.read(bytes.b, 4);
  const unsigned int num2 = bytes.u32;
  bool swap_user_item = false;
  if (num1 < num2) {
    swap_user_item = true;
  }
  if (!swap_user_item) {
    num_users_ = num1;
    num_items_ = num2;
  } else {
    num_items_ = num1;
    num_users_ = num2;
  }
  bytes.u64 = 0;
  file.read(bytes.b, 8);
  const uint64_t num_records = bytes.u64;
  record_vector_.reserve(num_records);
  // Get the data in the COO format
  for (int64_t i = 0; i < num_records; ++i) {
    int user_id = 0;
    int item_id = 0;
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const int id1 = bytes.u32 - 1;
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const int id2 = bytes.u32 - 1;
    if (!swap_user_item) {
      user_id = id1;
      item_id = id2;
    } else {
      item_id = id1;
      user_id = id2;
    }
    bytes.u64 = 0;
    file.read(bytes.b, 4);
    const float rating = bytes.f32;
    record_vector_.push_back(Record(user_id, item_id, rating));
  }
  file.close();
  return 0;
}

int Dataset::Shuffle() {
  logger_->Debug(stderr, "Shuffling records...\n");
  std::random_shuffle(record_vector_.begin(), record_vector_.end());
  logger_->Debug(stderr, "Records shuffled\n");
  return 0;
}

int Dataset::ComputeBlockSizes() {
  user_block_size_ = (num_users_ + mpi_info_->comm_size - 1) /
      mpi_info_->comm_size;
  item_block_size_ = (num_items_ + mpi_info_->comm_size - 1) /
      mpi_info_->comm_size;
  return 0;
}

int Dataset::AssignNewUserIds(const std::vector<int>& new_id_vector) {
  for (auto& r : record_vector_) {
    r.user_id = new_id_vector[r.user_id];
  }
  return 0;
}

int Dataset::AssignNewItemIds(const std::vector<int>& new_id_vector) {
  for (auto& r : record_vector_) {
    r.item_id = new_id_vector[r.item_id];
  }
  return 0;
}

int Dataset::GenerateLocalBlocks() {
  record_block_vector_.assign(mpi_info_->comm_size, std::vector<Record>());
  for (const auto& r : record_vector_) {
    int user_block_id = r.user_id / user_block_size_;
    if (user_block_id != mpi_info_->comm_rank) {
      continue;
    }
    int item_block_id = r.item_id / item_block_size_;
    record_block_vector_[item_block_id].push_back(r);
  }
  for (int i = 0; i < record_block_vector_.size(); ++i) {
    logger_->Debug("Dataset '%s' Block %d has %zu records\n",
                   name_.c_str(), i, record_block_vector_[i].size());
  }
  return 0;
}

int Dataset::ShuffleLocalBlocks() {
  logger_->Debug(stderr, "Shuffling local blocks...\n");
  for (auto& b : record_block_vector_) {
    std::random_shuffle(b.begin(), b.end());
  }
  logger_->Debug(stderr, "Local blocks shuffled\n");
  return 0;
}

int Dataset::GenerateLocalBlocksCoo() {
  user_id_block_vector_.assign(mpi_info_->comm_size, std::vector<int>());
  item_id_block_vector_.assign(mpi_info_->comm_size, std::vector<int>());
  rating_block_vector_.assign(mpi_info_->comm_size, std::vector<float>());
  for (int block_id = 0; block_id < record_block_vector_.size(); ++block_id) {
    for (const auto& record : record_block_vector_[block_id]) {
      user_id_block_vector_[block_id].push_back(record.user_id);
      item_id_block_vector_[block_id].push_back(record.item_id);
      rating_block_vector_[block_id].push_back(record.rating);
    }
  }
  return 0;
}

int Dataset::FlattenLocalBlocks() {
  for (int block_id = 0; block_id < mpi_info_->comm_size; ++block_id) {
    record_block_1d_vector_.insert(record_block_1d_vector_.end(),
                                   record_block_vector_[block_id].begin(),
                                   record_block_vector_[block_id].end());
  }
  return 0;
}

int Dataset::FlattenLocalBlocksCoo() {
  for (int block_id = 0; block_id < user_id_block_vector_.size(); ++block_id) {
    block_record_id_base_vector_.push_back(user_id_block_1d_vector_.size());
    user_id_block_1d_vector_.insert(user_id_block_1d_vector_.end(),
                                    user_id_block_vector_[block_id].begin(),
                                    user_id_block_vector_[block_id].end());
    item_id_block_1d_vector_.insert(item_id_block_1d_vector_.end(),
                                    item_id_block_vector_[block_id].begin(),
                                    item_id_block_vector_[block_id].end());
    rating_block_1d_vector_.insert(rating_block_1d_vector_.end(),
                                   rating_block_vector_[block_id].begin(),
                                   rating_block_vector_[block_id].end());
    block_num_record_vector_.push_back(user_id_block_vector_[block_id].size());
  }
  return 0;
}

int Dataset::AllocateGpuMemory() {
  cudaError_t e = cudaSuccess;
  e = cudaMalloc(&user_id_block_1d_vector_dev_,
                 user_id_block_1d_vector_.size() *
                     sizeof user_id_block_1d_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&item_id_block_1d_vector_dev_,
                 item_id_block_1d_vector_.size() *
                     sizeof item_id_block_1d_vector_[0]);
  logger_->CheckCudaError(e);
  e = cudaMalloc(&rating_block_1d_vector_dev_,
                 rating_block_1d_vector_.size() *
                     sizeof rating_block_1d_vector_[0]);
  logger_->CheckCudaError(e);
  return 0;
}

int Dataset::CopyToGpu() {
  cudaError_t e = cudaSuccess;
  e = cudaMemcpy(user_id_block_1d_vector_dev_, user_id_block_1d_vector_.data(),
                 user_id_block_1d_vector_.size() *
                     sizeof user_id_block_1d_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(item_id_block_1d_vector_dev_, item_id_block_1d_vector_.data(),
                 item_id_block_1d_vector_.size() *
                     sizeof item_id_block_1d_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  e = cudaMemcpy(rating_block_1d_vector_dev_, rating_block_1d_vector_.data(),
                 rating_block_1d_vector_.size() *
                     sizeof rating_block_1d_vector_[0],
                 cudaMemcpyHostToDevice);
  logger_->CheckCudaError(e);
  return 0;
}

int Dataset::ComputeNumItemsEffective() {
  std::vector<bool> has_rating_vector(num_items_, false);
  for (const auto& r : record_vector_) {
    has_rating_vector[r.item_id] = true;
  }
  num_items_effective_ = 0;
  for (const auto& b : has_rating_vector) {
    if (b) {
      ++num_items_effective_;
    }
  }
  logger_->Debug("Number of effective items = %d\n", num_items_effective_);
  return 0;
}

