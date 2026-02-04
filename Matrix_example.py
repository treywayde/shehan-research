import numpy as np
import pandas as pd
from matrix_data import matrix_4 # type: ignore

# This imports a graph function, to make graphing more simple and take up less space 
from matrix_data import graph    # type: ignore


##----------Variables to be changed-------------#
threshold_percentage = 20
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
initial_matrix = Z

#----------------------------------------------------------------------------------
# This function finds the edges of an object within a matrix, by using the magnitude of the 
# gradient to find edges. Returns a boolean matrix. 
def edgeDetection(inputMatrix):
    # Finds the gradient of the x and y
    matrix_y, matrix_x = np.gradient(inputMatrix)  # order is rows(y), cols(x)

    # Finds the magnitude of the 2D gradients, to give an overall gradient
    unrounded_gradient_magnitude = np.sqrt(matrix_x**2 + matrix_y**2)
    gradient_magnitude = np.round((unrounded_gradient_magnitude), 10)
    print("Gradient: \n", (gradient_magnitude),"\n")

    # Finds the maximum gradient value in the matrix 
    maximum_gradient_value = np.max(gradient_magnitude)
    print("Gradient Max Value: \n", maximum_gradient_value,"\n")

    # Sets a threshold of gradient values, for edge detection
    threshold = np.round(threshold_percentage * 0.01 * maximum_gradient_value, 5)
    print("Current Gradient Threshold: \n", threshold, "\n")

    # Creates a boolean matrix, with True to represent edges, 
    # and False to represent non-edges
    edges_boolean = gradient_magnitude >= threshold 
    print("Edge matrix in boolean format: \n", edges_boolean, "\n")

    # Converts the boolean matrix into an integer matrix 
    edges_numeric_output = edges_boolean.astype(int)
    print("Edge matrix in binary format: \n", edges_numeric_output, "\n")
    return edges_numeric_output
#----------------------------------------------------------------------------------

edges_numeric = edgeDetection(initial_matrix)

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
centroid_cols = avg_edges_columns
centroid_rows = avg_edges_rows

#-------------------------------------------------------------------------------------------
def findCenterLine(inputEdgesMatrix, inputInitialMatrix):

    # This finds the total width of the matrix
    total_width = len(inputEdgesMatrix[1])
    print("Total width of matrix: \n", total_width)

    # This creates an identical matrix, filled with zeros, to identify the centerline
    centerline = np.zeros_like(inputEdgesMatrix, dtype=float) 

    # Finds the centerline, by taking vertical slices of the shape, and finding the middle between 
    # the uppermost edge and the lowermost edge of that slice
    col = 0
    for column in range(total_width):
        # This grabs a slice from the edges_numeric matrix
        vert_slice = inputEdgesMatrix[:, column]

        # This finds all of the locations of any edges on that slice 
        locations_vert_slice = np.where(vert_slice == 1)[0]

        # If there are edges identified in a slice, it determines the central Y value in the shape
        # Then it adds a 1 in the centerline matrix at that exact index 
        if locations_vert_slice.size != 0:
            location_y = (locations_vert_slice[0] + locations_vert_slice[-1])/2
            centerline[int(np.round(location_y)), col] = 1
        col+=1

    # This graphs the initial matrix, then overlays the centerline over it in red 
    # graph(inputInitialMatrix, "centerline", centerline)
    
    return centerline
    #---------------------------------------------------------------------------------------

centerline = findCenterLine(edges_numeric, initial_matrix)

# creates an array of indices where the centerline binary matrix is one
centerline_location = np.argwhere(centerline==1)
print("Centerline Location: \n", centerline_location, "\n")


# This pulls out all of the rows and columns into a separate array, then determines how 
# many cells there are that are identified as the centerline
CL_rows = centerline_location[:,0]
CL_cols = centerline_location[:,1]
num_of_CL_cells = CL_cols.shape[0]
print("Centerline Rows: \n", CL_rows, "\n")
print("Centerline Columns: \n", CL_cols, "\n")
print("Number of Centerline cells: \n", num_of_CL_cells, "\n")


# This finds the relative rise/run for the centerline, two cells at a time. Run is not included, 
# because it is assumed to be 1 between each cell. Along with this, we filter out some of the first 
# values and last values, because the centerline is skewed along the edges of a turned shape
percent_cut = 15
num_cut_cells = int(np.round(num_of_CL_cells * percent_cut * 0.01))
print("Number of cut cells: \n", num_cut_cells, "\n")

# This for-loop goes from the bottom bound to top bound
slope = 0
for num in range(num_cut_cells, num_of_CL_cells - num_cut_cells):
    slope += (CL_rows[num +1] - CL_rows[num]) / (num_of_CL_cells - 2 * num_cut_cells)
print("Slope: \n", slope, "\n")

# Calculates the degrees of the slope 
angle = np.arctan(slope)
degree = angle * 180/np.pi
print("Degree: \n", degree, "\n")


# Makes a blank matrix the same size as the original
cols_length = len(initial_matrix)
rows_length = len(initial_matrix[0])

rotated_matrix = np.zeros_like(initial_matrix, dtype=float)
avg_value = np.mean(initial_matrix)
threshold_for_rotation = 0.15 * avg_value

# For each cell in the initial matrix, it finds where that cell should be when rotated. So, this
# does not change any values, or modify them, it simply moves them to where they would be. 
for i in range(cols_length):
    for j in range(rows_length):
        if initial_matrix[i,j] >= threshold_for_rotation:

            # Finds the location of any cell relative to the centroid, as if the centroid was at (0,0)
            init_cols = i - centroid_cols
            init_rows = j - centroid_rows

            # Finds the new location of that cell, in reference to the centroid
            new_cols = init_cols * np.cos(angle) - init_rows * np.sin(angle)
            new_rows = init_cols * np.sin(angle) + init_rows * np.cos(angle)

            # Takes away the influence of the centroid, and finds the location on the matrix
            i_new = int(np.round(new_cols + centroid_cols))
            j_new = int(np.round(new_rows + centroid_rows))

            # If the location is within the matrix bounds, it adds it to the rotated matrix
            if 0 <= i_new < cols_length and 0 <= j_new < rows_length:
                    rotated_matrix[i_new][j_new] = initial_matrix[i][j]

# graph(rotated_matrix, "Rotated Matrix")

# Runs the edge detection function on the rotated matrix 
rotated_matrix_edges = edgeDetection(rotated_matrix); 
# graph(rotated_matrix_edges, "Rotated Edges")
rotated_edges_location = np.argwhere(rotated_matrix_edges==1)
print("Location matrix of the rotated edges: \n", rotated_edges_location[:data_shown], "......\n")

# Finds the all of the values of the columns and rows of the edge locations 
rotated_edges_columns = rotated_edges_location[:,1]  
rotated_edges_rows = rotated_edges_location[:,0]
print("Location of the rotated edges columns max and min: \n", np.max(rotated_edges_columns), 
      np.min(rotated_edges_columns), "\n")
print("Location of the rotated edges rows max and min: \n", np.max(rotated_edges_rows), 
      np.min(rotated_edges_rows), "\n")

# Cuts out most of the blank space around the shape, just by finding max and min of 
# the locations of edge cells 
actual_shape = rotated_matrix[np.min(rotated_edges_rows): np.max(rotated_edges_rows), 
    np.min(rotated_edges_columns): np.max(rotated_edges_columns)]
# graph(actual_shape, "Shape Matrix")

# Sets the size of the desired matrix 
final_columns = 5
final_rows = 3

# Finds the width and height of each new cell 
cell_width = len(actual_shape[0]) /final_columns 
cell_height = len(actual_shape) / final_rows

# Makes a blank matrix to fill with the average values
sliced_final_shape = np.zeros([final_rows, final_columns], dtype = None)

# For-loop math
current_width = 0
current_height = 0
for i in range(final_columns):
    for j in range(final_rows):

       # Pulls out a small matrix slice from the big matrix 
        temp_shape = actual_shape[int(current_height): int(current_height) + int(np.ceil(cell_height)),
                                  int(current_width): int(current_width) + int(np.ceil(cell_width))]
        
        # If any values are less than 5% of the average, replace them with the average value. This ensures 
        # that any zeroes on the edges don't affect the outcome.
        temp_shape[temp_shape < avg_value*0.05] = avg_value

        # Fills in the value in the final matrix 
        sliced_final_shape[j, i] = np.round(temp_shape.mean(),3)

        # For-loop math 
        current_height += cell_height
    current_height = 0
    current_width += cell_width


# Prints out the final shape 
print(sliced_final_shape)





