import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

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
        self.images[camera_name] = cv_image

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

            # Check if we have all 4 images
            if len(images) == 4:
                # Construct the big image based on camera positions
                big_image = construct_big_image(images)
                
                # Create object map from big_image
                object_map = create_object_map(big_image)
                
                # Save object map as PGM and YAML files
                save_object_map(object_map, 'object_map')

                cv2.imshow("Stitched Image", big_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

def construct_big_image(images):
    # Extract images
    image1 = images['Camera 1']
    image2 = images['Camera 2']
    image3 = images['Camera 3']
    image4 = images['Camera 4']

    # Get dimensions of each image
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    h3, w3, _ = image3.shape
    h4, w4, _ = image4.shape

    # Determine dimensions of the stitched image
    max_height = max(h1 + h3, h2)
    max_width = max(w2 + w4, w1)

    # Create an empty canvas for the stitched image
    stitched_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    # Place each image onto the stitched image canvas
    # Bottom left (camera4)
    stitched_image[max_height-h4:max_height, 0:w4] = image4

    # Bottom right (camera1)
    stitched_image[max_height-h1:max_height, max_width-w1:max_width] = image1

    # Above camera1 (camera3)
    stitched_image[max_height-h1-h3:max_height-h1, max_width-w1:max_width] = image3

    # Above camera2 and to the left of camera3 (camera2)
    stitched_image[max_height-h2-h4:max_height-h4, 0:w2] = image2

    return stitched_image

def create_object_map(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    object_map = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Invert the object map (flip color coding)
    object_map = cv2.bitwise_not(object_map)
    
    return object_map


def save_object_map(object_map, base_filename):
    # Save as PGM file
    pgm_filename = base_filename + '.pgm'
    cv2.imwrite(pgm_filename, object_map)

    # Save as YAML file (metadata)
    yaml_filename = base_filename + '.yaml'
    with open(yaml_filename, 'w') as yaml_file:
        yaml_content = "image: {}.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196".format(base_filename)
        yaml_file.write(yaml_content)

if __name__ == '__main__':
    main()

