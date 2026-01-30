import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


# Variables to be changed
fileName = "SL_strip_a.csv"


# -----Reads in the File for Pandas Modifications------------------
df = pd.read_csv(fileName, header=None)  # no header
Z = df.to_numpy(dtype=float)
Z = np.nan_to_num(Z, nan=0.0)  # Converts NaN → 0
Z[Z < 0] = 0     #Turns every negative number into a 0 
#-------------------------------------------------------------------



# Calculates the change over the x axis
dx = Z[1:-1, 2:] - Z[1:-1, :-2]
# Z[1:-1, 2:] = rows from 1 to -1, starting from column 2 to the end
# Z[1:-1, :-2] = rows from 1 to -1, starting from beginning column and going to the second to last

# Calculates the change over y axis
dy = Z[2:, 1:-1] - Z[:-2, 1:-1]
# Z[1:-1, 2:] = rows from second to the end, columns from 1 to -1
# Z[1:-1, :-2] = rows from beginning to second to last, columns from 1 to -1

# Calculates the gradient, essentially the steepness 
gradient = np.sqrt(dx**2 + dy**2)


# Boolean mask of edges
threshold = 0.6 * np.max(gradient)  # edges are top 40% of gradients
edges = gradient >= threshold 

# Creates a matrix the same size as "Z"
edge_mask = np.zeros_like(Z, dtype=bool)

# Replaces the inside values with the smaller matrix we made called "edges"
edge_mask[1:-1, 1:-1] = edges

# Sets it to a binary format, 1 if True, 0 if False
edge_numeric = edge_mask.astype(int)
#--------------------------------------------------------------------------



#--------Basic plotting, overlays the actual shape with the edges--------------
plt.figure(figsize=(8, 6))

# Base matrix (grayscale)
plt.imshow(Z, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(Z, 99))
plt.colorbar(label="Z value")  # shows values

# Edge overlay (red)
plt.imshow(edge_mask, cmap="Reds", alpha=0.6, origin="lower")

plt.title("Matrix with Edge Overlay")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------


# Suppose edge_mask is True for object pixels
y, x = np.where(edge_mask)  # row, col of edges

# Center the coordinates
coords = np.vstack([x, y]).T
coords_centered = coords - coords.mean(axis=0)

# PCA using SVD
U, S, Vt = np.linalg.svd(coords_centered)
angle = np.arctan2(Vt[0,1], Vt[0,0])  # angle of the first principal axis
print("Angle (radians):", angle)
print("Angle (degrees):", np.degrees(angle))

from scipy.ndimage import rotate

# Rotate by -angle (to deskew)
Z_rotated = rotate(Z, angle=np.degrees(angle), reshape=True, order=1)  # bilinear interpolation

plt.figure(figsize=(8, 6))

# Base matrix (grayscale)
plt.imshow(Z_rotated, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(Z, 99))
plt.colorbar(label="Z value")  # shows values

plt.title("Rotated")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.tight_layout()
plt.show()


