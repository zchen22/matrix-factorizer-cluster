#ifndef MPI_INFO_H_
#define MPI_INFO_H_

// MPI header
#include <mpi.h>

struct MpiInfo {
 public:
  MpiInfo(const int num_processes_per_node);
  ~MpiInfo();
  int Load();
 public:
  int world_rank;
  int world_size;
  MPI_Comm comm;
  int comm_rank;
  int comm_size;
  // Cluster-specific data
  const int num_processes_per_node;
};

#endif

