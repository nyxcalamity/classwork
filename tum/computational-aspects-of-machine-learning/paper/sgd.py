for (x_t,y_t) in data_set:
	loss_fn = f(w, x_t, y_t)
	# compute gradient
	d_loss_fn_wrt_w = ...
	w -= gamma * d_loss_fn_wrt_w
	if <stopping condition is met>:
		return w
