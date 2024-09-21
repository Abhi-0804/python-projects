import numpy as np
import cv2

# Paths for models and input
prototxt_path = "models/colorization_deploy_v2.prototxt"
model_path = "models/colorization_release_v2.caffemodel"
kernel_path = 'models/pts_in_hull.npy'
image_path = r"C:\Users\abhis\OneDrive\Desktop\colorization project\images\flower.jpg"  # Use absolute path

# Load the pre-trained network and kernel points
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Reshape and set points for the network layers
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the black-and-white image
bw_image = cv2.imread(image_path)

# Check if the image was loaded properly
if bw_image is None:
    print(f"Error: Unable to load image at {image_path}. Please check the file path.")
    exit()

# Normalize the input image and convert to LAB color space
normalized = bw_image.astype("float32") / 255.0
lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

# Resize the L channel to the input size expected by the network (224x224)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50  # Mean-centering the L channel

# Pass the L channel to the network
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the predicted ab channels to the original image size
ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

# Combine original L channel with predicted ab channels
L = cv2.split(lab)[0]  # Original L channel of the input image
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# Convert LAB to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)

# Rescale colorized image back to range [0, 255]
colorized = (255.0 * np.clip(colorized, 0, 1)).astype("uint8")

# Create named windows
cv2.namedWindow("Black and White Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Colorized Image", cv2.WINDOW_NORMAL)

# Resize the window to a custom size (e.g., 500x500 pixels)
cv2.resizeWindow("Black and White Image", 500, 500)  # You can change this value
cv2.resizeWindow("Colorized Image", 500, 500)

# Display the images
cv2.imshow("Black and White Image", bw_image)
cv2.imshow("Colorized Image", colorized)

# Wait for keypress to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
