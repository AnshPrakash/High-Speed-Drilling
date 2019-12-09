import numpy as np 
  
# curve-fit() function imported from scipy 
from scipy.optimize import curve_fit 
from matplotlib import pyplot as plt 
  


# numpy.linspace with the given arguments 
# produce an array of 40 numbers between 0 
# and 10, both inclusive 
x = np.array(list(np.linspace(0, 10, num = 40))  + list(np.linspace(20, 30, num = 40)))
  
  
# y is another array which stores 3.45 times 
# the sine of (values in x) * 1.334.  
# The random.normal() draws random sample 
# from normal (Gaussian) distribution to make 
# them scatter across the base line 
y =  21*x - 8.32 + np.random.normal(size = 80)*10
  
# Test function with coefficients as parameters 
def test(x, a, b): 
  return (a*x + b)
  
# curve_fit() function takes the test-function 
# x-data and y-data as argument and returns  
# the coefficients a and b in param and 
# the estimated covariance of param in param_cov
param, param_cov = curve_fit(test, x, y) 
  
  
print("funcion coefficients:") 
print(param) 
print("Covariance of coefficients:") 
print(param_cov)
  
# ans stores the new y-data according to  
# the coefficients given by curve-fit() function 
ans = test(x,param[0],param[1])
# print(ans)
# ans = (param[0]*(np.sin(param[1]*x))) 
  
'''Below 4 lines can be un-commented for plotting results  
using matplotlib as shown in the first example. '''
  
plt.plot(x, y, 'o', color ='red', label ="data") 
plt.plot(x, ans, '--', color ='blue', label ="optimized data") 
plt.legend() 
plt.show() 