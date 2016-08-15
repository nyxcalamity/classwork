source("utils.R")

# Q1
cost <- 3
price <- 12
demand <- 87
order <- 100
profit <- price*demand-order*cost
profit

# Q2
x <- rep(0.2, 5)
y <- c(10, 20, 40, 60, 80)
xm <- sum(x*y)
xm

# Q3
st <- sqrt(sum((y-xm)^2/5))
st

# Q4
d <- c(78, 66, 95, 87, 57, 97, 114, 44, 74, 85, 95, 88, 69, 86, 50, 70, 90, 43, 61, 80)
dm <- mean(d)
dm
st <- sd(d)
st

# Q5


d <- c(1, 39, 19, 5, 97, 44, 49, 95, 46, 56, 3, 90, 2, 19, 66, 48, 11, 92, 99, 86)
idx <- 6:20
mse <- sum((d[idx]-SMA(d, idx))^2)/length(idx)
mse

# Q6
idx <- 4:20
mape <- 100*sum(abs(d[idx]-SMA(d, idx, 3))/d[idx])/length(idx)
mape

# Q7
1000+1000/sqrt(2500)

# Q8
1433*1.8