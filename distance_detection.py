import cv2
import numpy as np

def create_depth_map(img_left, img_right, calibration_matrix, baseline):
    # Convert images to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    # Stereo matching
    stereo = cv2.StereoBM_create(numDisparities=160, blockSize=7)
    disparity = stereo.compute(gray_left, gray_right)
    
    # Convert disparity to depth in meters
    depth_map = calibration_matrix[0, 0] * baseline / disparity
    
    return depth_map

if __name__ == "__main__":
    # Load stereo images
    img_left = cv2.imread('im0.png')
    img_right = cv2.imread('im1.png')
   
    
    # Calibration parameters

    fx = fy = 1733.74
    cx = 792.27
    cy = 541.89
    calibration_matrix = np.array([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0,  0,  1]])
    
  
    

    baseline = 0.53662  # baseline in meters
    
    # Create depth map
    depth_map = create_depth_map(img_left, img_right, calibration_matrix, baseline)
    
    # Display depth map
    cv2.imshow("Depth Map", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
