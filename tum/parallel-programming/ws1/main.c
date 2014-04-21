#include <time.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pi_series.h"
#include "helper.h"

static int digits_terms, pi_precision = 15;

void perror_exit(const char *str, const char *name)
{
	fprintf(stderr, "String \"%s\" is not a valid positive integer!\n"
			        "\n"
					"Usage: %s [NUM_TERMS] [NUM_THREADS]\n"
			        "If NUM_TERMS is not given default value will be used.\n"
					"NUM_THREADS has no effect for sequential application version\n",
					str, name);

	exit(EXIT_FAILURE);
}

void print_pi(long terms, double pi)
{
	printf("%*ld  %.*lf  %.*lf\n", digits_terms, terms, pi_precision, pi, pi_precision, fabs(M_PI - pi));
}

int main(int argc, char *argv[])
{
	double pi;

	long num_terms = 1000, num_threads = 1;

	if (argc > 1)
		if ((num_terms = strtol(argv[1], NULL, 0)) <= 0)
			perror_exit(argv[1], argv[0]);

	if (argc > 2)
		if ((num_threads = strtol(argv[2], NULL, 0)) == 0)
			perror_exit(argv[2], argv[0]);

	if(num_terms < num_threads){
		fprintf(stderr, "NUM_TERMS must be bigger then NUM_THREADS!\n");
		exit(EXIT_FAILURE);
	}

	digits_terms = snprintf(NULL, 0, "%ld", num_terms);
	digits_terms = digits_terms < 5 ? 5 : digits_terms;

	printf("\nApproximating PI with %ld Terms and %ld Threads\n\n", num_terms, num_threads);

	printf("%*s  %-*s   %-*s\n", digits_terms, "Terms",
			                    pi_precision+1, "Approximation",
			                    pi_precision+1, "Absolute Difference");

	for (int i = 1; i <= 10 ; ++i){
		pi = pi_series(i, num_threads);
		print_pi(i, pi);
	}

	printf("%*s\n", digits_terms, "...");

	struct timespec begin, end;

	clock_gettime(CLOCK_REALTIME, &begin);
	pi = pi_series(num_terms, num_threads);
	clock_gettime(CLOCK_REALTIME, &end);

	print_pi(num_terms, pi);

	char pi_str[pi_precision + 4];
	snprintf(pi_str, pi_precision + 3, "%.*lf", pi_precision, pi);

	printf("\nCorrect Digits: %d\n", str_cmatch(STR(M_PI), pi_str));

	printf("\nProcessing Time: %.3lf seconds\n", ts_to_double(ts_diff(begin, end)));

	return 0;
}
