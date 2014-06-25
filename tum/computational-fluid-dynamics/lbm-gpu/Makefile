# --------------------------------------------------------------------------------------------------
SOURCES=src/initialization.c src/visualization.c src/boundary.c src/collision.c src/streaming.c \
    src/compute_cell_values.c src/helper.c src/main.c
SOURCES_CU=src/collision_gpu.cu src/gpu_utils.cu
EXECUTABLE=build/lbm-sim
SRC_DIR=src
BUILD_DIR=build
COMPUTE_CAPABILITY=21
# --------------------------------------------------------------------------------------------------
#Compiler command
CC=g++
CC_CU=nvcc
#Compiler flags
CFLAGS=-g -Wall -ofast -funroll-loops #-Werror -pedantic
CFLAGS_CU=-g --ptxas-options=-v -arch=sm_$(COMPUTE_CAPABILITY)
# Linker flags
LDFLAGS=-lm
LDFLAGS_CU=-lcuda -lcudart
# --------------------------------------------------------------------------------------------------
OBJECTS=$(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJECTS_CU=$(SOURCES_CU:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

all: clean $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(OBJECTS_CU)
	$(CC) $(OBJECTS_CU) $(OBJECTS) -o $@ $(LDFLAGS) $(LDFLAGS_CU)

clean:
	rm -f $(OBJECTS) $(OBJECTS_CU) $(EXECUTABLE)
	rm -f img/*.vtk
	
$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

$(OBJECTS_CU): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.cu
	$(CC_CU) $(CFLAGS_CU) -c $< -o $@ $(LDFLAGS_CU)