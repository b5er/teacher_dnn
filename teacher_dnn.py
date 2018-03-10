from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp, array, random, dot, square
import os



# 216, 351, 330

'''
1. Alan Sussman
2. Nelson Padua-Perez
3. Jandelyn Plane

4. Clyde Kruskal
5. Evan Golub

6. Micahel Hicks
7. Jeffrey Foster
8. Chau-Wen Tseng
9. Alan Herman
'''

# our umd Ratings, Failing Metric, Difficulty Level

teacher_set_inputs = array([
       [.6, .1, 2],
       [.76, .07, 2],
       [.6, .14, 2],
       [.6, .07, 3],
       [.4, .1, 3],
       [.9, .03, 3],
       [.9, .03, 3],
       [.64, .05, 3],
       [.42, .15, 3]
])

# 1 = good
# 0 = bad
good_bad_outputs = array([[1, 1, 0, 0, 0, 1, 1, 0, 0]]).T

# 421 - Intro to AI
# Donald Perlis
should_teacher = array([.5, .05, 4])

def plotting():
	fig = plt.figure()
	a = fig.add_subplot(111, projection='3d')
	for i in range(len(teacher_set_inputs)):
		color = 'r'
		if good_bad_outputs[i] == 0:
			color = 'b'
		a.scatter([teacher_set_inputs[i][0]], [teacher_set_inputs[i][1]], [teacher_set_inputs[i][2]], c=color)
	a.scatter([should_teacher[0]], [should_teacher[1]], [should_teacher[1]], c='black')
	plt.show()

plotting()


# start Neural Network build

# ------------ Define Activation Function ----------------

# The Sigmoid function, which describes an S shaped curve.
# We pass the weighted sum of the inputs through this function to
# normalise them between 0 and 1.
def sigmoid(x):
    return 1 / (1 + exp(-x))

# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def deriv_sigmoid(x):
    return x * (1 - x)


# Seed random number generator
random.seed(1)

# ----------------- Initializing weights -----------------
# number of input, number of neurons
hidden_layer_w = 2 * random.random((3, 4)) - 1
# weights
print("input to hidden weights")
print(hidden_layer_w)

output_layer_w = 2 * random.random((4, 1)) - 1
# weights
print("hidden to output weights")
print(output_layer_w)



# ---------------------- Train ---------------------------

iterations = 60000
cost = []

for iteration in range(iterations):

	 hidden_layer_output = sigmoid(dot(teacher_set_inputs, hidden_layer_w))
	 output_layer_output = sigmoid(dot(hidden_layer_output, output_layer_w))



	 # ----------------- Back Propagation ----------------------------

	 # start from output layer and work yourself back to hidden layer
	 # weights.

	 # ---------- Output Layer -------------
	 # compute error of output layer. y_hat - y
	 output_layer_error = good_bad_outputs - output_layer_output
	 cost.append(output_layer_error[0])
	 output_layer_delta = output_layer_error * deriv_sigmoid(output_layer_output)

	 # ---------- Hidden Layer -------------
	 # compute error of hidden layer, goal is to see how hidden layer
	 # contributed to the error of output layer.
	 hidden_layer_error = output_layer_delta.dot(output_layer_w.T)
	 hidden_layer_delta = hidden_layer_error * deriv_sigmoid(hidden_layer_output)

	 # ---------- Adjust Weights -----------
	 # compute how much to adjust weights
	 hidden_adjust = teacher_set_inputs.T.dot(hidden_layer_delta)
	 output_adjust = hidden_layer_output.T.dot(output_layer_delta)

	 # adjust weights
	 hidden_layer_w += hidden_adjust
	 output_layer_w += output_adjust



# ------------------- Predict -----------------------------
hidden_layer_output = sigmoid(dot(should_teacher, hidden_layer_w))
output_layer_output = sigmoid(dot(hidden_layer_output, output_layer_w))



# fig = plt.plot(cost)
# plt.show()

def rate_professor(pred):
	if pred < .5:
		os.system("say Terrible")
	else:
		os.system("say Great")

rate_professor(output_layer_output)


print(output_layer_output)

