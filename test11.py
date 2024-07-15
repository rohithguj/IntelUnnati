import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Assuming you have some constant like CAMERA_POSITIONS defined
CAMERA_POSITIONS = [
    [0, 0],
    [0, 100],
    [100, 0],
    [100, 100]
]
NUM_OF_CAMS = 4

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.bridge = CvBridge()
        self.images = {}  # Dictionary to store images

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

        # Warp image to top-down view
        warped_image = self.warp_image_to_top_down(cv_image)
        self.images[camera_name] = warped_image

        # Check if we have images from all cameras
        if all(cam_name in self.images for cam_name in ['Camera 1', 'Camera 2', 'Camera 3', 'Camera 4']):
            # Stitch images from all cameras
            stitched_image = self.stitch_images()

            if stitched_image is not None:
                # Display the final stitched image
                cv2.imshow("Stitched Image", stitched_image)
                cv2.waitKey(1)

                # Save the final stitched image
                output_filename = 'stitched_image.png'
                cv2.imwrite(output_filename, stitched_image)
                print(f"Stitched image saved as {output_filename}")

    def warp_image_to_top_down(self, camera_image):
        # Example parameters
        H = 5  # Camera height in meters
        FOV = 90  # Field of view in degrees
        Rx, Ry = camera_image.shape[1], camera_image.shape[0]  # Camera resolution in pixels (horizontal, vertical)

        # Assuming intrinsic matrix K is known
        K = np.array([[1000, 0, Rx/2],
                      [0, 1000, Ry/2],
                      [0, 0, 1]])

        # Compute transformation matrix T
        focal_length = Rx / (2 * np.tan(np.radians(FOV / 2)))
        T = np.array([[focal_length, 0, Rx/2],
                      [0, focal_length, Ry/2],
                      [0, 0, 1]])

        # Initialize top-down view image
        top_down_view = np.zeros((Ry, Rx, 3), dtype=np.uint8)

        # Generate top-down view
        for y in range(Ry):
            for x in range(Rx):
                pixel_camera = np.array([x, y, 1])
                pixel_top_down = np.dot(T, pixel_camera)
                pixel_top_down = pixel_top_down.astype(int)

                if 0 <= pixel_top_down[0] < Rx and 0 <= pixel_top_down[1] < Ry:
                    top_down_view[y, x] = camera_image[pixel_top_down[1], pixel_top_down[0]]

        return top_down_view

    def stitch_images(self):
        # Extract images from self.images dictionary
        image1 = self.images.get('Camera 1', None)
        image2 = self.images.get('Camera 2', None)
        image3 = self.images.get('Camera 3', None)
        image4 = self.images.get('Camera 4', None)

        if image1 is None or image2 is None or image3 is None or image4 is None:
            self.get_logger().warn("Images for one or more cameras not found.")
            return None

        # Warp images to top-down view (if not already warped)
        warped_image1 = self.images['Camera 1']
        warped_image2 = self.images['Camera 2']
        warped_image3 = self.images['Camera 3']
        warped_image4 = self.images['Camera 4']

        # Determine dimensions of the stitched image
        h1, w1, _ = warped_image1.shape
        h2, w2, _ = warped_image2.shape
        h3, w3, _ = warped_image3.shape
        h4, w4, _ = warped_image4.shape

        max_height = max(h1, h2,

