# Include files
SOURCES=initLB.c visualLB.c boundary.c collision.c streaming.c computeCellValues.c main.c helper.c

# Compiler
# --------
CC=mpicc

CFLAGS=-g -Werror -pedantic -Wall -ofast -funroll-loops

# Linker flags
# ------------
LDFLAGS=-lm

OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=lbsim

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
	rm -f img/*.vtk

$(OBJECTS): %.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)