Assignment 4 - A 'cat' like utility

The source code consist of 3 files:

- ring_buffer.h
	A ring buffer / circular buffer implementation, that has some nice functions
	to make it easier to parallelize. ring_get_uncommited() and ring_put_uncommited()
	do not change the shared variable count. ring_get_count() and ring_put_count()
	read count and return the minimum readable / writable number of chars and block_size.
	ring_get_commit() and ring_put_commit() make the changes visible to the other
	thread by writing count.
	
- ring_buffer.c
	The implementation of the ring_buffer.c. Hopefully without bugs. But, I cannot
	guarantee this.
	
- main.c
	Reads by default char by char from stdin and writes them to stdout. You can
	set the buffer_size and the block_size. These parameters effect only the parallel
	version. fill_level makes only sense with pthread_conditional wait, to trigger
	a signal the producer that the buffer was emptied o consumer to signal new
	elements in the buffer. You can also provide 2 parameters and read from a file and
	write the output to a file. Be careful, you can easily fill up your hard disk.
	
	The code is already arranged in such a way, that it should be clear the producer
	is executed by thread 0 and the consumer by thread 1. You have to create two
	threads and passing all the necessary data to the threads. One important thing
	is the termination condition. It has to be passed from one thread to the other.
	
This assignment is probably the hardest one. You have to take care to do it right.
I would recommend a mutex only version first to parallelize it and to make it work.
Only then you should try to use cond_wait + mutex to avoid unnecessary polling.

How to test your code?

This time there is no makefile. Build it with

gcc -std=gnu99 -O2 main.c ring_buffer.c -o cat_par

Everything you write to stdin should come out unchanged of stdout. You could use some
text files test it.

$ cat readme.txt | ./cat_par

These parameters are accepted
$ ./car_pat buffer_size block_size fill_level file_in [file_out]

If you leave out file_out, it will be printed to stdout. 

To compare the speed of both version you could use:

$ ./cat_par 65536 512 0.75 /dev/zero | dd of=/dev/null
^C167137+0 records in
167136+0 records out
85573632 bytes (86 MB) copied, 4.14317 s, 20.7 MB/s

The implementation is quite slow, but it is simple. The first attempt I've implemented
had around 1,8 GB/s, but it was to complex. This is also the reason for the delay.
It was quite hard to make this assignment easier, without making it stupid.

The deadline is extended to next Friday instead of Tuesday. We will have the opportunity
to talk about your ideas for the solutions on Tuesday.

The next assignment will be easy and based on OpenMP. Sorry for the delay. Write me an
email if you find bugs in the code.