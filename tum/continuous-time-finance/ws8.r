#!/usr/bin/Rscript
# Task 1
#-------------------------------------------------------------------------------
# Initial data
#-------------------------------------------------------------------------------
S0=1; r=0.05; sigma=0.2; T=2; K=0.8;
#-------------------------------------------------------------------------------
Call <- function(S0,r, sigma, T, K){
  d1=(log(S0/K)+T*(r+(sigma^2)/2))/(sigma*sqrt(T))
  d2=d1-sigma*sqrt(T)
  C=S0*pnorm(d1)-K*exp(-r*T)*pnorm(d2)
  return (C)
}

ImpliedVola <- function(S0, r, T, K, C){
  vola=uniroot(function(sigma) Call(S0, r, sigma, T, K) - C, c(0,5))
  return (vola$root)
}

C=Call(S0,r,sigma,T,K)
C
ImpliedVola(S0,r,T,K,C)
#-------------------------------------------------------------------------------
Data=read.csv("data/option_values.csv",header=F,sep=";")
Maturities=as.numeric(Data[1,]/12); Interests=as.numeric(Data[2,])

S0=2890.62; K=c(2700, 2800, 2900, 3000, 3100, 3200, 3300)
vola=matrix(rep(0,42), 7, 6)
for(i in 1:7)
  for(j in 1:6)
    vola[i, j] = ImpliedVola(S0, Interests[j], Maturities[j], K[i], Data[i+2,j])

persp(K,Maturities,vola,theta=60,col="green")