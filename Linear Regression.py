import numpy as n
from sklearn.linear_model import LinearRegression as lr

x = n.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = n.array([5, 20, 14, 32, 22, 38])

# print(x)
# print(y)

print('')

model = lr().fit(x, y)
r_sq = model.score(x, y)
print('coeff of determinant: ', r_sq)
print('intercep: ', model.intercept_)
print('slope: ', model.coef_)

y_pred = model.predict(x)
print('prediction: ', y_pred)

nm = lr().fit(x, y.reshape((-1, 1)))
print('')
print('new model')
print('intercep: ', nm.intercept_)
print('slope: ', nm.coef_)
