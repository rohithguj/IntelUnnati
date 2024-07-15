import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
FOCAL_LENGTH = 800  # Assuming focal length in pixels
CX = 640
CY = 480

camera_matrix = np.array([[FOCAL_LENGTH, 0, CX], 
                          [0, FOCAL_LENGTH, CY], 
                          [0, 0, 1]], np.float32) 
                          
# Define the distortion coefficients 
dist_coeffs = np.zeros((5, 1), np.float32) 

NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
FOCAL_LENGTH = 1.0
CAMERA_POSITIONS = [
    [-5, -2, CAMERA_HEIGHT_IN_METERS],
    [-5, 3, CAMERA_HEIGHT_IN_METERS],
    [1, -2, CAMERA_HEIGHT_IN_METERS],
    [1, 3, CAMERA_HEIGHT_IN_METERS]
]

verList = []

# Populate verList and sort based on element[1] at the same time
verList = sorted(
    (
        sorted(
            [index for index, element in enumerate(CAMERA_POSITIONS) if element[0] == x],
            key=lambda idx: -CAMERA_POSITIONS[idx][1]  # Sort by y coordinate in descending order
        )
        for x in sorted(set(element[0] for element in CAMERA_POSITIONS), reverse=True)  # Sort x values in reverse order
    ),
    key=lambda sublist: -CAMERA_POSITIONS[sublist[0]][0]  # Sort by x coordinate in descending order
)

print(verList)

class ImageStitcher:
    def __init__(self):
        pass

    def stitch_images_locally(self, image1, image2, overlap_percent=0.4, direction='horizontal'):
        # Ensure images are grayscale
        if len(image1.shape) > 2:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if len(image2.shape) > 2:
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate overlap width or height based on percentage
        if direction == 'horizontal':
            overlap_size = int(image1.shape[1] * overlap_percent)
            left_image = image1[:, -overlap_size:]
            right_image = image2[:, :overlap_size]
            blended_region = self.blend_images(left_image, right_image)
            stitched_image = np.hstack((image1[:, :-overlap_size], blended_region, image2[:, overlap_size:]))
        elif direction == 'vertical':
            overlap_size = int(image1.shape[0] * overlap_percent)
            top_image = image1[-overlap_size:, :]
            bottom_image = image2[:overlap_size, :]
            blended_region = self.blend_images(top_image, bottom_image)
            stitched_image = np.vstack((image1[:-overlap_size, :], blended_region, image2[overlap_size:, :]))
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")

        return stitched_image

    def blend_images(self, image1, image2):
        # Simple blending using averaging
        return cv2.addWeighted(image1, 0.5, image2, 0.5, 0)


class ImageListener(Node):

    def __init__(self):
        super().__init__('image_listener')
        self.bridge = CvBridge()
        self.images = {}  # Dictionary to store images
        self.stitcher = ImageStitcher()  # Initialize ImageStitcher instance

        # Create subscriptions for each camera
        self.subscription1 = self.create_subscription(Image, '/overhead_camera/overhead_camera1/image_raw', self.listener_callback1, 10)
        self.subscription2 = self.create_subscription(Image, '/overhead_camera/overhead_camera2/image_raw', self.listener_callback2, 10)
        self.subscription3 = self.create_subscription(Image, '/overhead_camera/overhead_camera3/image_raw', self.listener_callback3, 10)
        self.subscription4 = self.create_subscription(Image, '/overhead_camera/overhead_camera4/image_raw', self.listener_callback4, 10)

    def listener_callback1(self, msg):
        self.process_image(msg, 'Camera 1')

    def listener_callback2(self, msg):
        self.process_image(msg, 'Camera 2')

    def listener_callback3(self, msg):
        self.process_image(msg, 'Camera 3')

    def listener_callback4(self, msg):
        self.process_image(msg, 'Camera 4')


    def process_image(self, msg, camera_name):
        self.get_logger().info(f'Receiving image from {camera_name}')
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Assuming you have 3D points to project (replace with actual data)
        # For example, let's project a single 3D point [10, 20, 30]
        x, y, z = 10, 20, 30
        points_3d = np.array([[[x, y, z]]], np.float32)

        # Define rotation and translation vectors
        rvec = np.zeros((3, 1), np.float32)
        tvec = np.zeros((3, 1), np.float32)

        # Project 3D point to 2D image plane
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)

        # Display the 2D point (for demonstration)
        print(f"Projected 2D Point for {camera_name}: {points_2d}")

        # Further processing to create a 2D map can be done here
        # For example, draw circles or lines at the projected points on the image

        # Apply Gaussian blur to reduce noise (example)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Perform watershed segmentation (example)
        markers = self.perform_watershed_segmentation(blurred_image)

        # Draw segmented regions and projected points on the map image
        map_image = self.draw_segmented_regions(cv_image, markers, points_2d)

        # Example: Display the map image (you can modify this part as needed)
        cv2.imshow("Map Image", map_image)
        cv2.waitKey(1)
        return map_image

    def perform_watershed_segmentation(self, image):
        # Ensure input image is in grayscale
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gradient calculation (Sobel)
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        gradient = cv2.magnitude(gradient_x, gradient_y)

        # Normalize gradient to [0, 255]
        gradient = np.uint8(255 * gradient / np.max(gradient))

        # Ensure gradient is converted to grayscale
        if len(gradient.shape) > 2:
            gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)

        # Threshold the gradient to obtain markers for watershed segmentation
        _, markers = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological operations to improve markers
        markers = cv2.morphologyEx(markers, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        # Convert image to color (3-channel) if it's grayscale
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) < 3 else image

        # Convert markers to 32-bit integer single-channel
        markers = np.int32(markers)

        # Perform watershed segmentation
        markers = cv2.watershed(image_color, markers)

        return markers

    def draw_segmented_regions(self, image, markers, points_2d=None):
        # Create a blank map image
        map_image = np.zeros_like(image)

        # Draw segmented regions based on markers (example)
        for marker in np.unique(markers):
            if marker <= 0:
                continue
            mask = np.zeros_like(markers, dtype=np.uint8)
            mask[markers == marker] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(map_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

        # Draw projected 2D points (example)
        if points_2d is not None:
            for point in points_2d:
                cv2.circle(map_image, tuple(point[0].astype(int)), 5, (0, 255, 0), -1)  # Green circle

        return map_image
         
    def vertStitch(self, stitchList):
        stitched_images = []

        for i in stitchList:
            stitched_row = None
            for j in i:
                camera_name = f'Camera {j + 1}'
                gray_image = self.images[camera_name]['gray']
                if stitched_row is None:
                    stitched_row = gray_image
                else:
                    # Stitch with overlap using ImageStitcher
                    stitched_row = self.stitcher.stitch_images_locally(stitched_row, gray_image, direction='horizontal')

            stitched_images.append(stitched_row)

        return stitched_images

    def horStitch(self, images):
        final_image = None
        for image in images:
            if final_image is None:
                final_image = image
            else:
                # Stitch vertically with overlap using ImageStitcher
                final_image = self.stitcher.stitch_images_locally(final_image, image, direction='vertical')

        return final_image

    def get_all_images(self):
        return self.images

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

            # Get images from all cameras
            images = node.get_all_images()

            # Check if we have all images
            if len(images) == NUM_OF_CAMS:
                # Perform horizontal stitching
                vertical_stitched = node.vertStitch(verList)

                # Perform vertical stitching
                big_image = node.horStitch(vertical_stitched)
                
                # Display stitched image
                cv2.imshow("Stitched Image", big_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

