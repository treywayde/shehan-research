import numpy as np
from matrix_data import matrix_2 # type: ignore

##----------Variables to be changed-------------#
threshold_percentage = 60
data_shown = 6
#----------------------------------------------#


# Initializes a Matrix, with 1s as the ground/nothing, and 3s as the object
initial_matrix = matrix_2


# Finds the gradient of the x and y
matrix_y, matrix_x = np.gradient(initial_matrix)  # order is rows(y), cols(x)
#print("Matrix_X:\n", Matrix_X, "\n")
#print("matrix_y:\n", matrix_y, "\n")


# Finds the magnitude of the 2D gradients, to give an overall gradient
unrounded_gradient_magnitude = np.sqrt(matrix_x**2 + matrix_y**2)
gradient_magnitude = np.round((unrounded_gradient_magnitude), 1)
print("Gradient: \n", (gradient_magnitude),"\n")


# Finds the maximum gradient value in the matrix 
maximum_gradient_value = np.max(gradient_magnitude)
print("Gradient Max Value: \n", maximum_gradient_value,"\n")


# Sets a threshold of gradient values, for edge detection
threshold = np.round((100 - threshold_percentage) * 0.01 * maximum_gradient_value, 3)
print("Current Gradient Threshold: \n", threshold, "\n")


# Creates a boolean matrix, with True to represent edges, 
# and False to represent non-edges
edges_boolean = gradient_magnitude >= threshold 
print("Edge matrix in boolean format: \n", edges_boolean, "\n")


# Converts the boolean matrix into an integer matrix 
edges_numeric = edges_boolean.astype(int)
print("Edge matrix in binary format: \n", edges_numeric, "\n")


# Finds the indices ([1,1] or [4,5]) where edges_numeric is equal to 1
# Essentially it finds the locations of all cells labeled as "edges", stores them in a 
# matrix of indices
ones_location = np.argwhere(edges_numeric==1)
print("Location matrix of the edges: \n", ones_location[:data_shown], "......\n")

#TODO rename ones_xvals and ones_yvals to be representative of columns and rows
# Extracts the x values and the y values, separates them into two different 1D matrices
ones_xvals = ones_location[:,1]  
ones_yvals = ones_location[:,0]  
ones_xvals_max, ones_xvals_min = ones_xvals.max(), ones_xvals.min()
ones_yvals_max, ones_yvals_min = ones_yvals.max(), ones_yvals.min()
print("X values of the edges: \n", ones_xvals[:data_shown], ".......\n")
print("Y values of the edges: \n", ones_yvals[:data_shown], ".......\n")
print("X value max, min: \n", ones_xvals_max, ", ",ones_xvals_min)
print("Y value max, min: \n", ones_yvals_max, ", ",ones_yvals_min)


# Calculates the average location of the edges, finding the centroid 
avg_ones_xvals = np.mean(ones_xvals)
avg_ones_yvals = np.mean(ones_yvals)
print("Average of the X values: \n", avg_ones_xvals, "\n")
print("Average of the Y values: \n", avg_ones_yvals, "\n")


# Temporarily displays the matrix and shows the centerpoint by labeling it with a 9,
# before changing it back to its original value
temp = edges_numeric[int(avg_ones_yvals), int(avg_ones_xvals)]
edges_numeric[int(avg_ones_yvals), int(avg_ones_xvals)] = 9
print("Identified centroid: \n", edges_numeric, "\n")
edges_numeric[int(avg_ones_yvals), int(avg_ones_xvals)] = temp


total_width = edges_numeric.shape[1]
print("Total width of matrix: \n", total_width)

centerline = np.zeros_like(edges_numeric, dtype=float) 


#TODO comment this
col = 0
for column in range(total_width):
    vert_slice = edges_numeric[:, column]
    locations_vert_slice = np.where(vert_slice == 1)[0]
    if locations_vert_slice.size != 0:
        location_y = (locations_vert_slice[0] + locations_vert_slice[-1])/2
        centerline[int(np.round(location_y)), col] = 1
    col+=1

print("Centerline is: \n", centerline)



