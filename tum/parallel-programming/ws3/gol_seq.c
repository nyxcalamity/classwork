#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int CalculateLiveNeighbours(unsigned char *grid, unsigned int dim_x, unsigned int dim_y, 
        unsigned int x, unsigned int y){
    unsigned char (*c_grid)[dim_x] = (unsigned char (*)[dim_x])grid;
    int live_neighbors=0;
    live_neighbors += ((y+1)<dim_y)                ? c_grid[y+1][x]   : 0;
    live_neighbors += ((x+1)<dim_x && (y+1)<dim_y) ? c_grid[y+1][x+1] : 0;
    live_neighbors += ((x+1)<dim_x)                ? c_grid[y][x+1]   : 0;
    live_neighbors += ((x+1)<dim_x && y!=0)        ? c_grid[y-1][x+1] : 0;
    live_neighbors += (y!=0)                       ? c_grid[y-1][x]   : 0;
    live_neighbors += (x!=0 && y!=0)               ? c_grid[y-1][x-1] : 0;
    live_neighbors += (x!=0)                       ? c_grid[y][x-1]   : 0;
    live_neighbors += (x!=0 && y+1<dim_y)          ? c_grid[y+1][x-1] : 0;
    return live_neighbors;
}

void evolve(unsigned char *grid_in, unsigned char *grid_out, unsigned int dim_x, unsigned int dim_y, unsigned int x, unsigned int y)
{
    unsigned char (*c_grid_in)[dim_x] = (unsigned char (*)[dim_x])grid_in;
    unsigned char (*c_grid_out)[dim_x] = (unsigned char (*)[dim_x])grid_in;
    int neighbors = CalculateLiveNeighbours(grid_in, dim_x, dim_y, x, y);
    
    if(c_grid_in[y][x]==0 && neighbors==3){ //check if at least 3 neighbors are alive
        c_grid_in[y][x]=1;
    }else if(!(neighbors==3 || neighbors==2)){ //check if 2 or 3 neighbors are alive
        c_grid_in[y][x]=0;
    }else{
        c_grid_out[y][x]=c_grid_in[y][x];
    }
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
