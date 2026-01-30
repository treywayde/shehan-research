import pandas as pd
import numpy as np

# Variables to be changed
fileName = "testing.csv"

new_length = 4
new_height = 3

# -----Reads in the File for Pandas Modifications------------------
df = pd.read_csv(fileName, header=None)  # no header
Z = df.to_numpy(dtype=float)
Z = np.nan_to_num(Z, nan=0.0)  # Converts NaN → 0
Z[Z < 0] = 0     #Turns every negative number into a 0 
#-------------------------------------------------------------------


def matrixSlice(matrix, des_height, des_length):

    # Finds the size of the inputted matrix
    height, length = matrix.shape

    # Makes a new matrix the correct size, rounding up or down to the nearent integer
    newMatrix = np.zeros((des_height, des_length), dtype=float)
    
    block_h = height // des_height
    block_w = length // des_length

    for i in range(round(des_height)):
        for j in range(round(des_length)):
            x0 = block_h*i
            x1 = block_h

            chunk = matrix[block_h*i:block_h*(i+1), block_w*j:block_w*(j+1)]

            newMatrix[i, j] = chunk.mean()

    print(newMatrix)
    
    return None

matrixSlice(Z, new_height, new_length)

