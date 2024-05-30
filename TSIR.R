library(tsBSS)
# NOT RUN {
n <- 10000

x1 <- arima.sim(n = n, list(ar = 0.2))
x2 <- arima.sim(n = n, list(ar = 0.3))
x3 <- arima.sim(n = n, list(ar = 0.2))
x4 <- arima.sim(n = n, list(ar = 0.2))
eps2 <- rnorm(n)
y <- 2*x1[1:(n)] + 3*x2[1:(n)] + eps2
X <- ((cbind(x1, x2, x3, x4))[1:n, ])

res1 <- tssdr(y, X, algorithm = "TSIR")
summ1 <- summary(res1, type = "alllag", thres = 0.8)
summ2 <- summary(res1, type = "alldir", thres = 0.8)
summ3 <- summary(res1, type = "rectangle", thres = 0.8)
summ4 <- summary(res1, type = "big", thres = 0.8)
coef(summ1)
coef(summ2)
coef(summ3)
coef(summ4)
