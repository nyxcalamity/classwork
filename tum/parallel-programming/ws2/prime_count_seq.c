#include <stdlib.h>
#include <math.h>
#include <stdio.h>

unsigned long prime_count(unsigned long a, unsigned long b, unsigned long num_threads, 
        unsigned long chunk_size){
    int count=0;
    if(a==1)
        a++;
    if(a==2){
        count++;a++;
    }
    for(int i=a; i<=b; i++){
        if(i%2!=0){
            int was_devisible_flag=0;
            for(int j=3; j<=ceil(sqrt(i)); j+=2){
                if(i%j==0 && !was_devisible_flag)
                    was_devisible_flag=1;
            }
            
            if(!was_devisible_flag)
                count++;
        }
    }
    return count;
}