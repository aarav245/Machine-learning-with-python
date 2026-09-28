import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6])
Y = np.array([30,40,50,60,70,80])

mean_x = np.mean(X)
mean_y = np.mean(Y)

num = 0
denom = 0

for i in range(len(X)):
    num += (X[i]-mean_x) * (Y[i]-mean_y)
    denom += (X[i]-mean_x)**2

m = num/denom

c = mean_y - m * mean_x

print("Slope (m): ", m)
print("Intercept (c): ", c)

predictedy = []

for value in X:
    prediction = m*value+c
    predictedy.append(prediction)

print("Predicted values:", predictedy)

errors = []

for i in range(len(Y)):
    error = Y[i]-predictedy[i]
    errors.append(error)

squarederrors = []

for error in errors:
    squarederrors.append(error**2)

mse = np.mean(squarederrors)
rmse = np.sqrt(mse)

print("MSE: ", mse)
print("RMSE: ",rmse)

hrs = 7

prediction = m*hrs+c
print("Predicted score for 7 hrs: ",prediction)

plt.scatter(X,Y,label = "Actual data")
plt.plot(X,predictedy,label = "Line of best fit")
plt.xlabel("Hours studied")
plt.ylabel("Exam scores")

plt.title("Linear regression - study hours versus exam score")
plt.legend()
plt.show()