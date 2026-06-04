import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os 

# Specify the path to your calibration images
images_path = './calibration_images/*.jpg'  # Adjust this path as needed

# Get a list of all calibration image file paths
images = sorted(glob.glob(images_path))

print(f"Found {len(images)} calibration images.")



# Define the chessboard size (number of inner corners)
chessboard_size = (10, 7)  # Adjust this based on your chessboard pattern

# Prepare object points
square_size_mm = 0.016
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size_mm
# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Iterate through all images
for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(np.shape(img))
    print(gray)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(ret)
    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)






ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




# Load a sample image for undistortion
sample_img = cv2.imread(images[0])
h, w = sample_img.shape[:2]

# Undistort the image
dst = cv2.undistort(sample_img, mtx, dist, None, mtx)





# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

print(f"Mean reprojection error: {mean_error:.5f} pixels")

# Calculate reprojection error for each point in each image
total_error = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = np.sum(np.abs(imgpoints[i] - imgpoints2)**2, axis=2).reshape(-1)
    total_error.extend(error)










def draw_camera(ax, R, t, scale=0.08):
    # Camera frame
    cam_points = np.array([
        [0, 0, 0],  # Camera center
        [1, 1, 2],  # Top-right corner
        [1, -1, 2], # Bottom-right corner
        [-1, -1, 2],# Bottom-left corner
        [-1, 1, 2]  # Top-left corner
    ]) * scale

    cam_points_world = (R @ cam_points.T + t.reshape(3, 1)).T

    # Plot camera frame
    # Base of the pyramid
    ax.plot(cam_points_world[[1,2,3,4,1], 0],
            cam_points_world[[1,2,3,4,1], 1],
            cam_points_world[[1,2,3,4,1], 2], 'r-')
    
    # Lines from camera center to corners
    for i in range(1, 5):
        ax.plot(cam_points_world[[0,i], 0],
                cam_points_world[[0,i], 1],
                cam_points_world[[0,i], 2], 'r-')

    # Plot camera center
    ax.scatter(cam_points_world[0, 0], cam_points_world[0, 1], cam_points_world[0, 2], color='r', s=50)

# Create 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot chessboard plane
X, Y, Z = objp[:,0], objp[:,1], objp[:,2]
ax.scatter(X, Y, Z, alpha=0.5)

# Plot camera poses
for i, (R, t) in enumerate(zip(rvecs, tvecs)):
    R = cv2.Rodrigues(R)[0]
    R = np.linalg.inv(R)
    t = -t
    
    draw_camera(ax, R, t)
    ax.text(t[0][0], t[1][0], t[2][0], f'{i+1}', fontsize=8)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Poses Relative to Chessboard')

# Set equal aspect ratio
ax.set_box_aspect((1, 1, 1))

# Get the limits of all axes
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
z_lim = ax.get_zlim()

# Find the range
max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0

# Find the center
mid_x = np.mean(x_lim)
mid_y = np.mean(y_lim)
mid_z = np.mean(z_lim)

# Set the limits for all axes
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()

# --- NEW SAVING LOGIC ---
# Define the folder and file name
output_folder = "first-part-result"
os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist
save_path = os.path.join(output_folder, "camera_poses_3d.png")

# Save the figure BEFORE calling plt.show()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\nPlot successfully saved to: {save_path}")
# ------------------------

# Show the plot
plt.show()

# Calculate and print camera position statistics
camera_positions = np.array([t.ravel() for t in tvecs])
mean_position = np.mean(camera_positions, axis=0)
std_position = np.std(camera_positions, axis=0)

print(f"Mean camera position: X={mean_position[0]:.2f}, Y={mean_position[1]:.2f}, Z={mean_position[2]:.2f}")
print(f"Standard deviation of camera positions: X={std_position[0]:.2f}, Y={std_position[1]:.2f}, Z={std_position[2]:.2f}")

# Interpretation of the results
print("\nInterpretation of Camera Pose Visualization:")
print("1. Each point represents a camera position for a calibration image.")
print("2. The chessboard plane is shown at Z=0.")
print("3. The spread of camera positions shows the variety of viewpoints used in calibration.")
print("4. Ideally, you want a good distribution of camera positions around the chessboard.")
print("5. If all points are clustered in one area, consider taking more diverse calibration images.")