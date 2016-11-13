#include "mpi-info.h"

// C++ headers
#include <cassert>
#include <cstdlib>

MpiInfo::MpiInfo(const int num_processes_per_node)
    : world_rank(-1), world_size(0), comm_rank(-1), comm_size(0),
      num_processes_per_node(num_processes_per_node) {
  int ret = MPI_SUCCESS;
  ret = MPI_Init(NULL, NULL);
  assert(ret == MPI_SUCCESS);
}

MpiInfo::~MpiInfo() {
  MPI_Finalize();
}

int MpiInfo::Load() {
  int ret = MPI_SUCCESS;
  // Get world info
  ret = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  assert(ret == MPI_SUCCESS);
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  assert(ret == MPI_SUCCESS);
  // Create communicator
  MPI_Group world_group;
  ret = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  assert(ret == MPI_SUCCESS);
  const int group_size = world_size / num_processes_per_node;
  int* ranks = NULL;
  ranks = (int*) calloc(group_size, sizeof(int));
  for (int i = 0; i < group_size; ++i) {
    ranks[i] = i * num_processes_per_node;
  }
  MPI_Group group;
  if (world_rank % num_processes_per_node == 0) {
    ret = MPI_Group_incl(world_group, group_size, ranks, &group);
    assert(ret == MPI_SUCCESS);
  } else {
    ret = MPI_Group_incl(world_group, group_size, ranks, &group);
    assert(ret == MPI_SUCCESS);
  }
  delete[] ranks;
  ret = MPI_Comm_create(MPI_COMM_WORLD, group, &comm);
  assert(ret == MPI_SUCCESS);
  // Get communicator info
  if (world_rank % num_processes_per_node == 0) {
    ret = MPI_Comm_size(comm, &comm_size);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Comm_rank(comm, &comm_rank);
    assert(ret == MPI_SUCCESS);
  } 
  return 0;
}

