import scipy
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.signal as signal
from google.colab import files
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.ndimage import zoom

def getIndexes(mask, targetH, targetW, offsetX=0, offsetY=0):
    indexes = np.zeros((targetH, targetW))

    maskH, maskW = mask.shape

    index_counter = 1
    for y in range(maskH):
        for x in range(maskW):
            if mask[y, x]:
                targetY, targetX = y + offsetY, x + offsetX
                # Check if the indices are within the bounds of the target image
                if 0 <= targetY < targetH and 0 <= targetX < targetW:
                    indexes[targetY, targetX] = index_counter
                    index_counter += 1

    return indexes

def getCoefficientMatrix(indexes):
    """
    Constructs the coefficient matrix (A in Ax=b)

    Args:
    - indexes: targetH * targetW, indexes of replacement area in target image

    Return:
    - A: N * N(N is max index), a matrix corresponds to laplacian kernel, 4 on the diagonal and -1 for each neighbor
    """
    # IMPLEMENT HERE
    # Since the coefficient matrix A is by nature sparse. Consider using scipy.sparse.csr_matrix to represent A to save space

    targetH, targetW = indexes.shape

    # N is the max index
    N = int(np.max(indexes))

    # Create an empty sparse matrix in LIL format (for easier insertions)
    from scipy.sparse import lil_matrix
    A = lil_matrix((N, N), dtype=np.int32)

    # Loop over each pixel in the target image
    for y in range(targetH):
        for x in range(targetW):
            if indexes[y, x] > 0:
                idx = int(indexes[y, x] - 1)  # Convert to 0-based index

                # Set the diagonal to 4
                A[idx, idx] = 4

                # set the value to -1
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < targetH and 0 <= nx < targetW and indexes[ny, nx] > 0:
                        n_idx = int(indexes[ny, nx] - 1)
                        A[idx, n_idx] = -1

    A = A.tocsr()

    return A

def getSolutionVect(indexes, source, target, offsetX, offsetY):
    """
    Constructs the target solution vector (b in Ax=b)

    Args:
    - indexes:  targetH * targetW, indexes of replacement area in target image
    - source, target: source and target image
    - offsetX, offsetY: int, offset of source image origin in the target image

    Returns:
    - solution vector b (for single channel)
    """
    # # IMPLEMENT HERE

    targetH, targetW = indexes.shape
    b = np.zeros(int(np.max(indexes)))

    # Loop over each pixel in the replacement area
    for y in range(targetH):
        for x in range(targetW):
            if indexes[y, x] > 0:
                idx = int(indexes[y, x] - 1)

                # 1. Laplacian part from the source image
                source_y, source_x = y - offsetY, x - offsetX
                laplacian = 4 * source[source_y, source_x]
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    sy, sx = source_y + dy, source_x + dx
                    if 0 <= sy < source.shape[0] and 0 <= sx < source.shape[1]:
                        laplacian -= source[sy, sx]

                b[idx] += laplacian

                # 2. Pixel part from the target image
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < targetH and 0 <= nx < targetW and indexes[ny, nx] == 0:
                        b[idx] += target[ny, nx]

    # 3. Already added the two parts together in the loops above

    return b

  def solveEqu(A, b):
    """
    Solve the equation Ax = b to get replacement pixels x in the replacement area
    Note: A is a sparse matrix, so we need to use corresponding function to solve it

    Args:
    - A: Laplacian coefficient matrix
    - b: target solution vector

    Returns:
    - x: solution of Ax = b
    """
    # IMPLEMENT HERE
    # you may find scipy.sparse.linalg.spsolve useful to solve equation
    x = spsolve(A, b)

    return x

  def reconstructImg(indexes, red, green, blue, target):
    """
    Reconstruct the target image with new red, green, blue channel values in the
    indexes area

    Args:
    - indexes: targetH * targetW, indexes of replacement area in target image
    - red, green, blue: 1 x N, three chanels for replacement pixel values
    - target: target image

    Returns:
    - resultImg: reconstructed target image with poisson editing
    """
    # # IMPLEMENT HERE

    # Initialize the result image with the target image
    #resultImg = np.copy(target)
    resultImg = np.copy(target)
    targetH, targetW = target.shape[:2]

    # 1. get nonzero component in indexes
    rows, cols = np.nonzero(indexes)

    # 2. stack three channels together with numpy dstack
    red = np.atleast_1d(red)
    green = np.atleast_1d(green)
    blue = np.atleast_1d(blue)
    RGB = np.dstack((red, green, blue)).reshape(-1, 3)

    # 3. copy new pixels in the indexes area to the target image
    for i in range(len(rows)):
        resultImg[rows[i], cols[i]] = RGB[i]

    return resultImg

    """
Function (do not modify)
"""
def seamlessCloningPoisson(sourceImg, targetImg, mask, offsetX, offsetY):
    """
    Wrapper function to put all steps together

    Args:
    - sourceImg, targetImg: source and targe image
    - mask: logical mask of source image
    - offsetX, offsetY: offset of source image origin in the target image

    Returns:
    - ResultImg: result image
    """
    # step 1: index replacement pixels
    indexes = getIndexes(mask, targetImg.shape[0], targetImg.shape[1], offsetX,
                         offsetY)
    # step 2: compute the Laplacian matrix A
    A = getCoefficientMatrix(indexes)

    # step 3: for each color channel, compute the solution vector b
    red, green, blue = [
        getSolutionVect(indexes, sourceImg[:, :, i], targetImg[:, :, i],
                        offsetX, offsetY).T for i in range(3)
    ]

    # step 4: solve for the equation Ax = b to get the new pixels in the replacement area
    new_red, new_green, new_blue = [
        solveEqu(A, channel)
        for channel in [red, green, blue]
    ]

    # step 5: reconstruct the image with new color channel
    resultImg = reconstructImg(indexes, new_red, new_green, new_blue,
                               targetImg)
    return resultImg

