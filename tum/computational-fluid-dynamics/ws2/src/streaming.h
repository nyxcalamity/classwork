#ifndef _STREAMING_H_
#define _STREAMING_H_

/** carries out the streaming step and writes the respective distribution functions from
 *  collideField to streamField.
 */
void doStreaming(double *collide_field, double *stream_field, int *flag_field, int xlength);
#endif