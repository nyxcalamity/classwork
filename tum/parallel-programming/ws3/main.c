#include <time.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gol.h"
#include "helper.h"

/* Use: gol_seq time_steps dim_x dim_y num_threads
   If no values are given default values will be used.
   num_threads has no effect on sequential application */

void print_gol(unsigned char *grid, unsigned int dim_x, unsigned int dim_y)
{
	unsigned char (*c_grid)[dim_x] = (unsigned char (*)[dim_x])grid;

	size_t size = sizeof(unsigned char) * dim_x + 4;

	unsigned char *row = malloc(size);
	if(row == NULL)
		exit(EXIT_FAILURE);

	memset(row, '-', size);
	row[0] = '+';
	row[size-3] = '+';
	row[size-2] = '\n';
	row[size-1] = '\0';

	printf("%s", row);

	row[0] = '|';
	row[size-3] = '|';

	for (int i = 0; i < dim_y; ++i) {
		for (int j = 0; j < dim_x; ++j) {
			if(c_grid[i][j] == 0)
				row[j+1] = ' ';
			else
				row[j+1] = '*';
		}
		printf("%s", row);
	}

	memset(row, '-', size);
	row[0] = '+';
	row[size-3] = '+';
	row[size-2] = '\n';
	row[size-1] = '\0';

	printf("%s", row);

	free(row);
}

void r_pentomino(unsigned char *grid, unsigned int dim_x, unsigned int dim_y, unsigned int x, unsigned int y)
{
	unsigned char (*c_grid)[dim_x] = (unsigned char (*)[dim_x])grid;

	c_grid[(y + dim_y - 1) % dim_y][(x + dim_x - 0) % dim_x] = 1;
	c_grid[(y + dim_y - 1) % dim_y][(x + dim_x + 1) % dim_x] = 1;

	c_grid[(y + dim_y - 0) % dim_y][(x + dim_x - 1) % dim_x] = 1;
	c_grid[(y + dim_y - 0) % dim_y][(x + dim_x - 0) % dim_x] = 1;

	c_grid[(y + dim_y + 1) % dim_y][(x + dim_x - 0) % dim_x] = 1;
}

int main(int argc, char *argv[])
{
	unsigned int dim_x = 80, dim_y = 40, time_steps = 80, num_threads = 3;

	if (argc > 1)
		time_steps = strtoul(argv[1], NULL, 0);

	if (argc > 2)
		dim_x = strtoul(argv[2], NULL, 0);

	if (argc > 3)
		dim_y = strtoul(argv[3], NULL, 0);

	if (argc > 4)
		num_threads = strtoul(argv[4], NULL, 0);

	if(dim_x < 9 || dim_y < 9)
	{
		printf("Invalid dim_x / dim_y!\n");
		exit(EXIT_FAILURE);
	}

	size_t size = sizeof(unsigned char) * dim_x * dim_y;
	unsigned char *grid = malloc(size);

	if(grid == NULL)
		exit(EXIT_FAILURE);

	memset(grid, 0, size);

	r_pentomino(grid, dim_x, dim_y, dim_x/2, dim_y/2);

	printf("\nGame of Life: time_steps = %u; dim_x = %u; dim_y = %u; threads = %u \n\n", time_steps, dim_x, dim_y, num_threads);

	print_gol(grid, dim_x, dim_y);

	printf("\n\n");

	struct timespec begin, end;

	clock_gettime(CLOCK_REALTIME, &begin);
	unsigned int living_cells = gol(grid, dim_x, dim_y, time_steps, num_threads);
	clock_gettime(CLOCK_REALTIME, &end);

	print_gol(grid, dim_x, dim_y);

	printf("Living Cells after %u time steps: %u\n", time_steps, living_cells);

	printf("\nProcessing Time: %.3lf seconds\n", ts_to_double(ts_diff(begin, end)));

	free(grid);

	return 0;
}
