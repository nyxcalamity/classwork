# --------------------------------------------------------------------------------------------------
#Source files
SOURCES=src/initLB.c src/visualLB.c src/boundary.c src/collision.c src/streaming.c \
    src/computeCellValues.c src/main.c src/helper.c
EXECUTABLE=lbsim
SRC_DIR=src
BUILD_DIR=build
# --------------------------------------------------------------------------------------------------
#Compiler command
CC=mpicc
#Compiler flags
CFLAGS=-g -Werror -Wall -ofast -funroll-loops #-pedantic
# Linker flags
LDFLAGS=-lm
# --------------------------------------------------------------------------------------------------
OBJECTS=$(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
	rm -f img/*.vtk

$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)