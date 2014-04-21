#ifndef HELPER_H_
#define HELPER_H_

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

int str_cmatch(const char* a, const char* b);
struct timespec ts_diff(struct timespec a, struct timespec b);
double ts_to_double(struct timespec time);

#endif /* HELPER_H_ */
