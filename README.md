# edx-unit1-asst

# Part One: Toy Linear Regression Problem 
The first part of the assignment will by a simple linear regression example. It will consist of several data points that you will need to run a standard linear regression on y vs x. You will then be asked to graph the points, and understand what the true relationship is. Once this is completed, you will also be adding features such as quadratic terms in order to make your model fit better. 

# Part Two: SysID Linear Regression 
The second part of the assignment will be a linear regression problem and system identification application. In this setup, we expect some model to behave according to x[t+1] = Ax[t] + bu[t]. Given a model using the Model class to query and collect data, you will be responsible for formatting your data collection and manipulating the data into a least squares problem. Finally, you will also be responsible to recover the parameters A and b that best fit the model. 

**Note**: There will be a Model class in part two of the assignment. It implements a forward method, given an input and potentially a state. If a state is inputted, the model will run from that state; otherwise, the model will run from its internal state. The forward method will output the next state. 
