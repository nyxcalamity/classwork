#include <time.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prime_count.h"
#include "helper.h"

static int max_digits;

/* Use: sum_sum_??? [A] [B] [NUM_THREADS] [CHUNK_SIZE]\n"
   If no values are given default values will be used.
   NUM_THREADS and CHUNK_SIZE have no effect on sequential
   application version */

void print_count(unsigned long a, unsigned long b, unsigned long sum)
{
	printf("[%*lu, %-*lu]   %-*lu\n", max_digits, a, max_digits, b, max_digits, sum);
}

int main(int argc, char *argv[])
{
	unsigned long a = 10000, b = 100000, num_threads = 3, chunk_size = 5, count = 0;

	if (argc > 1)
		a = strtoul(argv[1], NULL, 0);

	if (argc > 2)
		b= strtoul(argv[2], NULL, 0);

	if (argc > 3)
		num_threads = strtoul(argv[3], NULL, 0);

	if (argc > 4)
		chunk_size = strtoul(argv[4], NULL, 0);

	if(a > b)
	{
		printf("Invalid interval!\n");
		exit(EXIT_FAILURE);
	}

	max_digits = snprintf(NULL, 0, "%lu", b);
	max_digits = max_digits < 5 ? 5 : max_digits;

	printf("\nCounting Prime Numbers: threads = %lu chunk_size = %lu\n\n", num_threads, chunk_size);

	printf("   Interval\n");
	printf("[%*c, %-*c]   # Prime Numbers\n\n", max_digits, 'a', max_digits, 'b');

	unsigned long a_tmp = 1;

	for (int i = 10; i <= 10000 ; i*=10){
		count = prime_count(a_tmp, i, num_threads, chunk_size);
		print_count(a_tmp, i, count);
		a_tmp = i;
	}

	printf("%*s\n", max_digits, "...");

	struct timespec begin, end;

	clock_gettime(CLOCK_REALTIME, &begin);
	count = prime_count(a, b, num_threads, chunk_size);
	clock_gettime(CLOCK_REALTIME, &end);

	print_count(a, b, count);

	printf("\nProcessing Time: %.3lf seconds\n", ts_to_double(ts_diff(begin, end)));

	return 0;
}
