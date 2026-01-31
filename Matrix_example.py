import numpy as np
import pandas as pd
from matrix_data import matrix_4 # type: ignore

# This imports a graph function, to make graphing more simple and take up less space 
from matrix_data import graph    # type: ignore


##----------Variables to be changed-------------#
threshold_percentage = 60
data_shown = 6
fileName = "SL_strip_a.csv"
#----------------------------------------------#

# -----Reads in the File for Pandas Modifications------------------
df = pd.read_csv(fileName, header=None)  # no header
Z = df.to_numpy(dtype=float)
Z = np.nan_to_num(Z, nan=0.0)  # Converts NaN → 0
Z[Z < 0] = 0     #Turns every negative number into a 0 
#-------------------------------------------------------------------


# Initializes a Matrix, either from a CSV or from a predetermined data set
initial_matrix = matrix_4


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
edges_location = np.argwhere(edges_numeric==1)
print("Location matrix of the edges: \n", edges_location[:data_shown], "......\n")


# Extracts the x values and the y values, separates them into two different 1D matrices
edges_columns = edges_location[:,1]  
edges_rows = edges_location[:,0]  
print("X values of the edges: \n", edges_columns[:data_shown], ".......\n")
print("Y values of the edges: \n", edges_rows[:data_shown], ".......\n")


# Extracts the maximum and minimum values of the edge's columns and rows 
edges_columns_max, edges_columns_min = edges_columns.max(), edges_columns.min()
edges_rows_max, edges_rows_min = edges_rows.max(), edges_rows.min()
print("X value max, min: \n", edges_columns_max, ", ",edges_columns_min)
print("Y value max, min: \n", edges_rows_max, ", ",edges_rows_min)


# Calculates the average location of the edges, finding the centroid 
avg_edges_columns = np.mean(edges_columns)
avg_edges_rows = np.mean(edges_rows)
print("Average of the X values: \n", avg_edges_columns, "\n")
print("Average of the Y values: \n", avg_edges_rows, "\n")


# Temporarily displays the matrix and shows the centerpoint by labeling it with a 9,
# before changing it back to its original value
temp = edges_numeric[int(avg_edges_rows), int(avg_edges_columns)]
edges_numeric[int(avg_edges_rows), int(avg_edges_columns)] = 9
print("Identified centroid: \n", edges_numeric, "\n")
edges_numeric[int(avg_edges_rows), int(avg_edges_columns)] = temp

# This finds the total width of the matrix
total_width = edges_numeric.shape[1]
print("Total width of matrix: \n", total_width)


# This creates an identical matrix, filled with zeros, to identify the centerline
centerline = np.zeros_like(edges_numeric, dtype=float) 


# Finds the centerline, by taking vertical slices of the shape, and finding the middle between 
# the uppermost edge and the lowermost edge of that slice
col = 0
for column in range(total_width):
    # This grabs a slice from the edges_numeric matrix
    vert_slice = edges_numeric[:, column]

    # This finds all of the locations of any edges on that slice 
    locations_vert_slice = np.where(vert_slice == 1)[0]

    # If there are edges identified in a slice, it determines the central Y value in the shape
    # Then it adds a 1 in the centerline matrix at that exact index 
    if locations_vert_slice.size != 0:
        location_y = (locations_vert_slice[0] + locations_vert_slice[-1])/2
        centerline[int(np.round(location_y)), col] = 1
    col+=1


# This graphs the initial matrix, then overlays the centerline over it in red 
graph(initial_matrix, "centerline", centerline)




