#ifndef RING_BUFFER_H_
#define RING_BUFFER_H_

#include <stdlib.h>

typedef struct ring_buf
{
	size_t size;
	size_t count;
	size_t head;
	size_t tail;
	char elem[];
} ring_buf_t;

size_t ring_get_count(ring_buf_t *rb, size_t block_size);

size_t ring_put_count(ring_buf_t *rb, size_t block_size);

int ring_put_uncommited(ring_buf_t *rb, char c);

void ring_put_commit(ring_buf_t *rb, size_t count);

int ring_get_uncommited(ring_buf_t *rb, char *c);

void ring_get_commit(ring_buf_t *rb, size_t count);

#endif /* RING_BUFFER_H_ */
