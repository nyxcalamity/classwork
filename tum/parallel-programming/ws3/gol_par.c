#include <stdlib.h>
#include <string.h>

void evolve(unsigned char *grid_in, unsigned char *grid_out, unsigned int dim_x, unsigned int dim_y, unsigned int x, unsigned int y)
{

}

void swap(unsigned char **a, unsigned char **b)
{
	unsigned char *tmp = *a;
	*a = *b;
	*b = tmp;
}

unsigned int cells_alive(unsigned char *grid, unsigned int dim_x, unsigned int dim_y)
{
	unsigned char (*c_grid)[dim_x] = (unsigned char (*)[dim_x])grid;

	unsigned int cells = 0;

	for (int y = 0; y < dim_y; ++y)
	{
		for (int x = 0; x < dim_x; ++x)
		{
			cells += c_grid[y][x];
		}
	}

	return cells;
}

unsigned int gol(unsigned char *grid, unsigned int dim_x, unsigned int dim_y, unsigned int time_steps, unsigned int num_threads)
{
	unsigned char *grid_in, *grid_out, *grid_tmp;
	size_t size = sizeof(unsigned char) * dim_x * dim_y;

	grid_tmp = malloc(size);
	if(grid_tmp == NULL)
		exit(EXIT_FAILURE);

	memset(grid_tmp, 0, size);

	grid_in = grid;
	grid_out = grid_tmp;

	for (int t = 0; t < time_steps; ++t)
	{
		for (int y = 0; y < dim_y; ++y)
		{
			for (int x = 0; x < dim_x; ++x)
			{
				evolve(grid_in, grid_out, dim_x, dim_y, x, y);
			}
		}
		swap(&grid_in, &grid_out);
	}

	if(grid != grid_in)
		memcpy(grid, grid_in, size);

	free(grid_tmp);

	return cells_alive(grid, dim_x, dim_y);
}
