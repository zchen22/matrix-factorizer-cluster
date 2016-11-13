#include "configuration-set.h"

// C++ headers
#include <cassert>
#include <fstream>
#include <sstream>

ConfigurationSet::ConfigurationSet(MpiInfo* mpi_info, Logger* logger)
    : gd_mode(kGdModeAdapSgd), batch_size(1), num_features(128),
      learning_rate(0.1), regularization_factor(0.05),
      max_num_iterations(70), decomp_mode(kDecompModeFeature), lock(false),
      show_train_rmse(false), show_test_rmse(false), mpi_info(mpi_info),
      logger(logger) {
}

ConfigurationSet::~ConfigurationSet() {
}

int ConfigurationSet::Load(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    logger->Warning(stderr, "Config file '%s' not found\n", filename.c_str());
    logger->Warning(stderr, "Using default configurations\n");
    return 0;
  }
  logger->Info(stderr, "Reading config from file '%s'...\n", filename.c_str());
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream line_stream(line);
    std::string word;
    assert(line_stream >> word);
    if (word[0] == '%') {
      continue;
    }
    if (!word.compare("gd_mode")) {
      int v = 0;
      assert(line_stream >> v);
      gd_mode = static_cast<GdMode>(v);
    } else if (!word.compare("batch_size")) {
      assert(line_stream >> batch_size);
    } else if (!word.compare("num_features")) {
      assert(line_stream >> num_features);
    } else if (!word.compare("learning_rate")) {
      assert(line_stream >> learning_rate);
    } else if (!word.compare("regularization_factor")) {
      assert(line_stream >> regularization_factor);
    } else if (!word.compare("max_num_iterations")) {
      assert(line_stream >> max_num_iterations);
    } else if (!word.compare("decomp_mode")) {
      int v = 0;
      assert(line_stream >> v);
      decomp_mode = static_cast<DecompMode>(v);
    } else if (!word.compare("lock")) {
      assert(line_stream >> lock);
    } else if (!word.compare("show_train_rmse")) {
      assert(line_stream >> show_train_rmse);
    } else if (!word.compare("show_test_rmse")) {
      assert(line_stream >> show_test_rmse);
    } else {
      logger->Warning(stderr, "Unrecognized configuration '%s'\n",
                      word.c_str());
    }
  }
  file.close();
  return 0;
}

std::string ConfigurationSet::ToString() const {
  std::stringstream out;
  out << "gd_mode = " << gd_mode << " "
      << "batch_size = " << batch_size << " "
      << "num_features = " << num_features << " "
      << "learning_rate = " << learning_rate << " "
      << "regularization_factor = " << regularization_factor << " "
      << "max_num_iterations = " << max_num_iterations << " "
      << "decomp_mode = " << decomp_mode << " "
      << "lock = " << lock << " "
      << "show_train_rmse = " << show_train_rmse << " "
      << "show_test_rmse = " << show_test_rmse;
  return out.str();
}

