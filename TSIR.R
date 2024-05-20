library(tsBSS)
# NOT RUN {
n <- 10000
A <- matrix(rnorm(9), 4, 4)

x1 <- arima.sim(n = n, list(ar = 0.2))
x2 <- arima.sim(n = n, list(ar = 0.8))
x3 <- arima.sim(n = n, list(ar = 0.3, ma = -0.4))
x4 <- arima.sim(n = n, list(ar = 0.3, ma = -0.4))
eps2 <- rnorm(n - 1)
y <- 2*x1[1:(n - 1)] + eps2
X <- ((cbind(x1, x2, x3, x4))[2:n, ]) %*% t(A)

res1 <- tssdr(y, X, algorithm = "TSIR")
res1
summ1 <- summary(res1, type = "alllag", thres = 0.8)
summ1
plot(summ1)
head(components(summ1))
coef(summ1)
# Hybrid of TSIR and TSAVE. For TSIR part H = 10 and for TSAVE part H = 2.
#tssdr(y, X, algorithm = "TSSH", weight = 0.6, H = c(10, 2))
# }