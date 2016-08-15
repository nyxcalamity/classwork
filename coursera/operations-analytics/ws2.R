source("utils.R")

# Q1
prod.supply <- c(150, 130, 15)
prod.o <- c(0.1, 0.05, 0.03)
prod.b <- c(0.3, 0.1, 0.02)
prod.boxProfit <- c(5.0, 5.3)
prod.maxDemand <- c(250, 400)

# Q1a
demand <- c(250, 350)
profit <- sum(demand*prod.boxProfit)
supply <- demand[1]*prod.o+demand[2]*prod.b
sprintf("Profit: $%.2f", profit)
do.call(sprintf, c(list("Supply balance: %.1f %.1f %.1f"), prod.supply-supply))

# Q1b
sprintf("Juice requirements: %.2f", supply[3])

# Q1c
supplyBalance <- prod.supply - (prod.maxDemand[1]*prod.o + prod.maxDemand[2]*prod.b)
sprintf("No, because there's not enough juice supply.")

# Q1h
profit <- function(demand){
  return(-sum(demand*prod.boxProfit))
}
ui <- matrix(c(-prod.o[1], -prod.b[1], 
               -prod.o[2], -prod.b[2], 
               -prod.o[3], -prod.b[3], 
               -1, 0, 
               0, -1,
               1, 0, 
               0, 1), ncol = 2, byrow = TRUE)
ci <- c(-prod.supply[1], -prod.supply[2], -prod.supply[3], -prod.maxDemand[1], -prod.maxDemand[2], 
        0, 0)
opt <- constrOptim(c(1, 1), profit, NULL, ui, ci)
do.call(sprintf, c(list("Optimum allocation:  %.0f %.0f"), opt$par))
do.call(sprintf, c(list("Optimum value:  %.2f"), -opt$value))

# Q2
prod.stock <- c(3900, 3600)
prod.pricePerBox <- c(20, 32) 
trucks.capacities <- c(2700, 3100, 2800)
trucks.spoilRate <- matrix(c(5, 10, 4, 12, 3, 11), ncol = 2, byrow = TRUE)/100

# Q2b
order <- matrix(c(1300, 1200, 1300, 1200, 1300, 1200), ncol = 2, byrow = TRUE)
sprintf("Total profit: %.2f", sum(((1-trucks.spoilRate) * order) %*% prod.pricePerBox))

# Q2e
revenue <- function(order){
  return(-sum(((1-trucks.spoilRate) * matrix(order, ncol = 2, byrow = TRUE)) %*% prod.pricePerBox))
}

ui <- rbind(
  c(-1, -1, 0, 0, 0, 0),
  c(0, 0, -1, -1, 0, 0),
  c(0, 0, 0, 0, -1, -1),
  c(1, 0, 0, 0, 0, 0), 
  c(0, 1, 0, 0, 0, 0), 
  c(0, 0, 1, 0, 0, 0), 
  c(0, 0, 0, 1, 0, 0), 
  c(0, 0, 0, 0, 1, 0), 
  c(0, 0, 0, 0, 0, 1),
  c(-1, 0, -1, 0, -1, 0),
  c(0, -1, 0, -1, 0, -1))
ci <- c(-trucks.capacities, rep(0, 6), -prod.stock)
opt <- constrOptim(c(1, 1, 1, 1, 1, 1), revenue, NULL, ui, ci)

do.call(sprintf, c(list("Optimum allocation: %.0f %.0f, %.0f %.0f, %.0f %.0f"), opt$par))
do.call(sprintf, c(list("Optimum value:  %.2f"), -opt$value))

