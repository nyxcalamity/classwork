# --------------------------------------------------------------------------------------------------
#Source files
SOURCES=src/initialization.c src/visualization.c src/boundary.c src/collision.c src/streaming.c \
    src/compute_cell_values.c src/helper.c src/main.c
CUSOURCES=src/cuda_calls.cu
EXECUTABLE=build/lbm-sim
SRC_DIR=src
BUILD_DIR=build
# --------------------------------------------------------------------------------------------------
#Compiler command
CC=g++
CUCC=nvcc
#Compiler flags
CFLAGS=-g #-Werror -Wall -ofast -funroll-loops -pedantic
CUFLAGS=-arch=sm_30
# Linker flags
LDFLAGS=-lm
CULDFLAGS=-lcuda -lcudart
# --------------------------------------------------------------------------------------------------
OBJECTS=$(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
CUOBJECTS=$(CUSOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(CUOBJECTS)
	$(CC) $(CUOBJECTS) $(OBJECTS) -o $@ $(LDFLAGS) $(CULDFLAGS)

#$(EXECUTABLE): $(OBJECTS) 
#	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(CUOBJECTS) $(EXECUTABLE)
	rm -f img/*.vtk
	
$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

$(CUOBJECTS): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CUCC) $(CFLAGS) -c $< -o $@