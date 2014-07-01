# --------------------------------------------------------------------------------------------------
SRC_DIR=src
BUILD_DIR=build

SOURCES=
EXECUTABLE=
# --------------------------------------------------------------------------------------------------
#Compiler command
CC=g++
#Compiler flags
CFLAGS=-g -Wall -ofast -funroll-loops #-Werror -pedantic
# Linker flags
LDFLAGS=-lm
# --------------------------------------------------------------------------------------------------
OBJECTS=$(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

all: clean $(EXECUTABLE)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS) $(LDFLAGS_CU)

$(OBJECTS): $(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)
