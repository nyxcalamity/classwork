SMA <- function(series, idx, period=5){
  # TODO(me): utilize previous calculations for SMA instead of recalculating it for every index.
  # Computes simple moving average (SMA) on provided series for specified index (or a vector of 
  # indexes).
  sma <- rep(0, length(idx))
  j <- 1
  for (i in c(idx)){
    if(i > period)
      sma[j] <- sum(series[(i-period):(i-1)])/period
    j <- j+1
  }
  return(sma)
}