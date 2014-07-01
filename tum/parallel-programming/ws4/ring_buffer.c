#include "ring_buffer.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

size_t ring_get_count(ring_buf_t *rb, size_t block_size)
{
	return MIN(rb->count, block_size);
}

size_t ring_put_count(ring_buf_t *rb, size_t block_size)
{
	// size >= count
	return MIN(rb->size - rb->count, block_size);
}

int ring_put_uncommited(ring_buf_t *rb, char c)
{
    if (rb->count < rb->size)
    {
      rb->elem[rb->head] = c;
      rb->head = (rb->head + 1) % rb->size;

      return 0;
    }
    return -1;
}

// commit before calling ring_put_uncommited again
void ring_put_commit(ring_buf_t *rb, size_t count)
{
    rb->count += count;
}

int ring_get_uncommited(ring_buf_t *rb, char *c)
{
	if (rb->count > 0)
	{
		*c = rb->elem[rb->tail];
		rb->tail = (rb->tail + 1) % rb->size;

		return 0;
	}
	return -1;
}

// commit before calling ring_get_uncommited again
void ring_get_commit(ring_buf_t *rb, size_t count)
{
    rb->count -= count;
}
