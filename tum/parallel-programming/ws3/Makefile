NAME = gol
CC = gcc
CFLAGS = -std=gnu99 -pthread -fstrict-overflow -Werror -Wall -Wshadow -Wstrict-overflow=5 -pedantic
LDFLAGS = -lm -lpthread -lrt

#Compiler flags
#if mode variable is empty, setting release build mode
ifeq ($(mode),debug)
	CFLAGS += -ggdb
else
	mode = release
	CFLAGS += -O2
endif

SRC = $(wildcard *.c)

OBJ_SEQ = main.o $(NAME)_seq.o helper.o
OBJ_PAR = main.o $(NAME)_par.o helper.o
OBJ_SOL1 = main.o $(NAME)_par_sol1.o helper.o
OBJ_SOL2 = main.o $(NAME)_par_sol2.o helper.o
OBJ_UNIT = unit_test.o $(NAME)_par.o $(NAME)_ref.o helper.o
OBJ_COUNT = pthread_count.o $(NAME)_par.o
OBJ_PERF = performance_test.o $(NAME)_par.o helper.o

all: sequential parallel $(NAME)_par.o

sequential: $(NAME)_seq

parallel: $(NAME)_par

$(NAME)_seq : $(OBJ_SEQ)
	$(CC) -o $@ $(OBJ_SEQ) $(CFLAGS) $(LDFLAGS)

$(NAME)_par : $(OBJ_PAR)
	$(CC) -o $@ $(OBJ_PAR) $(CFLAGS) $(LDFLAGS)
	
$(NAME)_par_sol1 : $(OBJ_SOL1)
	$(CC) -o $@ $(OBJ_SOL1) $(CFLAGS) $(LDFLAGS)
	
$(NAME)_par_sol2 : $(OBJ_SOL2)
	$(CC) -o $@ $(OBJ_SOL2) $(CFLAGS) $(LDFLAGS)	

unit_test: $(OBJ_UNIT)
	$(CC) -o $@ $(OBJ_UNIT) $(CFLAGS) $(LDFLAGS)
	
pthread_count: $(OBJ_COUNT)
	$(CC) -o $@ $(OBJ_COUNT) $(CFLAGS) $(LDFLAGS) 
	
performance_test: $(OBJ_PERF)
	$(CC) -o $@ $(OBJ_PERF) $(CFLAGS) $(LDFLAGS)
 
%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

clean:
	rm -f $(NAME)_seq $(NAME)_par $(NAME)_par_sol1 $(NAME)_par_sol2 unit_test pthread_count performance_test *.o *.d
	
-include $(SRC:.c=.d)

.PHONY : all sequential parallel clean
