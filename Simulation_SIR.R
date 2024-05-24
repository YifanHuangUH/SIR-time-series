library(dr)

# Generate Data
set.seed(123)
n <- 10000

x1 <- arima.sim(n = n, list(ar = 0.2))
x2 <- arima.sim(n = n, list(ar = 0.2))
x3 <- arima.sim(n = n, list(ar = 0.3, ma = 0.4))
x4 <- arima.sim(n = n, list(ar = -0.4))

eps2 <- rnorm(n)
Y <- 2*x1[1:n] + 3*x2[1:n] + eps2
X <- ((cbind(x1, x2, x3, x4))[1:n, ])

# Apply SIR
result <- dr(Y ~ X, method = "sir", nslices = 50)

result