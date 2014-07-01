#include <stdio.h>
#include <stdlib.h>

#include "ring_buffer.h"

int main(int argc, char *argv[])
{
	size_t buffer_size = 65536, block_size = 512;
	double fill_level = 0.5;

	if (argc > 1)
		buffer_size = strtoul(argv[1], NULL, 0);

	if (argc > 2)
	{
		block_size = strtoul(argv[2], NULL, 0);
		if(buffer_size < block_size)
		{
			fprintf(stderr, "buffer_size must be bigger block_size\n");
			exit(EXIT_FAILURE);
		}
	}

	if(argc > 3)
	{
		fill_level = strtod(argv[3], NULL);
		if(fill_level >= 1.0 || fill_level <= 0.0)
		{
			fprintf(stderr, "fill_level must be between ]0.0 and 1.0[\n");
			exit(EXIT_FAILURE);
		}
	}

	if (argc > 4)
		if (freopen(argv[4], "r", stdin) == NULL)
		{
			perror("freopen()");
			fprintf(stderr, "Cannot open file: %s\n", argv[4]);
			exit(EXIT_FAILURE);
		}

	if (argc > 5)
		if (freopen(argv[5], "w", stdout) == NULL)
		{
			perror("freopen()");
			fprintf(stderr, "Cannot open file: %s\n", argv[5]);
			exit(EXIT_FAILURE);
		}

	ring_buf_t *rb = calloc(sizeof(*rb) + buffer_size, sizeof(char));
	if(rb == NULL)
	{
		perror("malloc()");
		exit(EXIT_FAILURE);
	}
	rb->size = buffer_size;

	size_t put_count = 0, get_count = 0, eof_flag = 0;

	for (;;)
	{
		/* Thread 0 - Producer */
		while ((put_count = ring_put_count(rb, block_size)) > 0 && !eof_flag)
		{
			for (int i = 0; i < put_count; ++i)
			{
				char c = fgetc(stdin);

				if(c == EOF)
				{
					eof_flag = 1;
					put_count -= put_count - i;
					break;
				}
				ring_put_uncommited(rb, c);
			}
			ring_put_commit(rb, put_count);
		}
		/* Thread 0 - Producer */

		/* Thread 1 - Consumer */
		while ((get_count = ring_get_count(rb, block_size)) > 0)
		{
			for (int i = 0; i < get_count; ++i)
			{
				char c;
				ring_get_uncommited(rb, &c);
				fputc(c, stdout);
			}
			ring_get_commit(rb, get_count);
		}
		/* Thread 1 - Consumer */

		// enable for debug: fflush(stdout);

		if(eof_flag == 1)
			break;
	}

	free(rb);

	return EXIT_SUCCESS;
}
