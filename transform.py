import cv2
import numpy as np

def order_points(pts):
    # initializing a 2 array of 4 rows and 2 cols
    # such that the 1st row has coor of top-left
    # 2nd row has coor of top-right
    # 3rd row has coor of bottom-right
    # 4th row has coor of bottom-left
    rect = np.zeros((4,2),dtype="float32")

    # finding the coordinates of the top-left and bottom-right
    # top-left will have smallest sum of x and y coor and 
    # bottom-right will have the largest sum of xand y coor
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # finding the coodinates of top-right and bottom-left
    # top-right will have the smallest difference of x and y coor
    # whereas the bottom-left will have the max diff of x and y coor
    diff = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

#%%
def four_pts_transform(image,pts):

    # we get the points of the roi of the image
    rect = order_points(pts)
    # we separate the points
    (tl,tr,br,bl) = rect

    # calculating the maxWidth of the image by applying the distance formula
    # distance formula sqrt((x1-x2)^2 + (y1-y2)^2)
    widthA = np.sqrt(((tr[0] - tl[0])**2) +  ((tr[1] - tl[1])**2))
    widthB = np.sqrt( ((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    maxWidth = max( int(widthA), int(widthB))

    # similarly calculating the maxHeight of the image
    heightA = np.sqrt( ((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2) )
    heightB = np.sqrt( ((tr[0] - br[0])**2) + ((tr[1] - br[1])**2) )
    maxHeight = max( int(heightA), int(heightB))

    # constructing a matrix with tl,tr,br,bl
    dst = np.array([
        [0, 0], # tl
        [maxWidth - 1, 0], # tr
        [maxWidth - 1, maxHeight - 1], # br
        [0, maxHeight - 1]], # bl 
        dtype="float32")
    
    # now using the cv2.perspectiveTransform to get the perspective matrix
    BirdView = cv2.getPerspectiveTransform(rect, dst)
    # passing the perspective matrix to warpPerspective to get the top-bottom view of image
    finalView = cv2.warpPerspective(image, BirdView, (maxWidth, maxHeight))

    return finalView