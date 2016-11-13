# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Files
CCSRCS = $(wildcard $(SRCDIR)/*.cc)
CUSRCS = $(wildcard $(SRCDIR)/*.cu)
OBJS = $(patsubst $(SRCDIR)/%.cc, $(OBJDIR)/%.o, $(CCSRCS)) \
       $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CUSRCS))
BIN = $(BINDIR)/mf

# Common
CUDA = /shared/apps/cuda7.5

# Compilers and options
MPICC = mpiCC
NVCC = $(CUDA)/bin/nvcc
CCFLAGS = -I$(CUDA)/include -std=c++11 -Wall
NVFLAGS = -arch=sm_35 -std=c++11 --use_fast_math
LDFLAGS = -std=c++11 -Wall
ifeq ($(DEBUG), 1)
  CCFLAGS += -g
  NVFLAGS += --ptxas-options=-v -g
  LDFLAGS += -g
else
  CCFLAGS += -O2
  NVFLAGS += -O2
  LDFLAGS += -O2
endif
LIBS = -L$(CUDA)/lib64 -lcudart -lm -lpthread

all: make_dirs $(BIN)

make_dirs:
	mkdir -p $(OBJDIR) $(BINDIR)

$(BIN): $(OBJS)
	$(MPICC) -o $@ $(LDFLAGS) $^ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(MPICC) -o $@ $(CCFLAGS) -c $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) -o $@ $(NVFLAGS) -c $<

.PHONY: clean
clean:
	rm -fr $(BINDIR) $(OBJDIR)

