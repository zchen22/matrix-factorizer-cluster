// C++ header
#include <unordered_map>

// MPI header
#include <mpi.h>

// Project header
#include "matrix-factorizer.h"

int ParseCommandArgs(const int argc, const char** argv,
                     std::unordered_map<std::string, std::string>& arg_map) {
  for (int i = 1; i < argc; ++i) {
    if (!strncmp(argv[i], "-t", strlen(argv[i]))) {
      arg_map["-t"] = argv[++i];
    } else if (!strncmp(argv[i], "-e", strlen(argv[i]))) {
      arg_map["-e"] = argv[++i];
    } else if (!strncmp(argv[i], "-c", strlen(argv[i]))) {
      arg_map["-c"] = argv[++i];
    } else {
      fprintf(stderr, "Unrecognized argument '%s'\n", argv[i]);
      exit(EXIT_FAILURE);
    }
  }
  if (arg_map.find("-t") == arg_map.end()) {
    fprintf(stderr, "Argument '-t' required!\n");
    exit(EXIT_FAILURE);
  }
  return 0;
}


int MatrixFactorDistrib(std::unordered_map<std::string, std::string>& arg_map) {
  MpiInfo mpi_info(48);
  mpi_info.Load();
  // Create one process per node (assuming one GPU per node)
  if (mpi_info.world_rank % mpi_info.num_processes_per_node == 0) {
    MatrixFactorizer mf(&mpi_info);
    mf.Setup(arg_map);
    mf.InitializeFeatures();
    mf.Preprocess();
    mf.AllocateGpuMemory();
    mf.CopyToGpu();
    mf.Sync();
    mf.Train();
    mf.DumpFeatures();
  }
  return 0;
}

int PrintHelp(const char* exec) {
  fprintf(stderr, "Usage: %s -t train-file [-e test-file] [-c config-file]"
              "\n\n", exec);
  fprintf(stderr, "-t train-file:\n");
  fprintf(stderr, "\tTrain data file in COO format.\n");
  fprintf(stderr, "\tEach line is a triplet (user-id, item-id, rating).\n");
  fprintf(stderr, "\tThe first line is a triplet "
          "(number-of-users, number-of-items, number-of-ratings).\n");
  fprintf(stderr, "\tComment lines start with '%%'s.\n");
  fprintf(stderr, "[-e test-file]:\n");
  fprintf(stderr, "\tTest data file in COO format.\n");
  fprintf(stderr, "\tEach line is a triplet (user-id, item-id, rating).\n");
  fprintf(stderr, "\tThe first line is a triplet "
          "(number-of-users, number-of-items, number-of-ratings).\n");
  fprintf(stderr, "\tComment lines start with '%%'s.\n");
  fprintf(stderr, "[-c config-file]:\n");
  fprintf(stderr, "\tConfiguration file.\n");
  fprintf(stderr, "\tComment lines start with '#'s.\n");
  return 0;
}

int main(const int argc, const char** argv) {
  if (argc >= 3) {
    std::unordered_map<std::string, std::string> arg_map;
    ParseCommandArgs(argc, argv, arg_map);
    MatrixFactorDistrib(arg_map);
    return 0;
  }
  PrintHelp(argv[0]);
  return 0;
}

