import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
FOCAL_LENGTH = 1.0
CAMERA_POSITIONS = [
    [-5, -2, CAMERA_HEIGHT_IN_METERS],
    [-5, -3, CAMERA_HEIGHT_IN_METERS],
    [1, -2, CAMERA_HEIGHT_IN_METERS],
    [1, -3, CAMERA_HEIGHT_IN_METERS]
]

horList = []
horVisit = []
stitched_images = []

# Populate horList and sort based on element[1] at the same time
horList = sorted(
    (
        [index for index, element in enumerate(CAMERA_POSITIONS) if element[1] == y]
        for y in sorted(set(element[1] for element in CAMERA_POSITIONS))
    ),
    key=lambda sublist: CAMERA_POSITIONS[sublist[0]][1]
)
''' 
    STITCH ORDER:
        1) Left -> Right [HORIZONTAL]
        2) Bottom -> Top [VERTICAL]
'''

print(horList)

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

        # Convert to grayscale
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        H, W = cv_image_gray.shape[:2]

        # Camera parameters
        camera_height = 8  # meters
        horizontal_fov = 60  # degrees

        # Calculate ground coverage
        G_w = 2 * camera_height * np.tan(np.radians(horizontal_fov) / 2)
        G_h = G_w * (H / W)  # Maintain aspect ratio

        # Warp the image to bird's eye view
        cv_image_birdseye = self.warp_image_to_bird_eye_view(cv_image_gray, G_w, G_h)

        # Store both color and grayscale images
        self.images[camera_name] = {
            'color': cv_image,
            'gray': cv_image_gray,
            'birdseye': cv_image_birdseye
        }

    def get_ground_coordinates(self, u, v, W, H, G_w, G_h):
        S_x = G_w / W
        S_y = G_h / H
    
        X = (u - W / 2) * S_x
        Y = (v - H / 2) * S_y
    
        return X, Y

    def warp_image_to_bird_eye_view(self, image, G_w, G_h):
        H, W = image.shape[:2]
        warped_image = np.zeros((H, W), dtype=np.uint8)
    
        for v in range(H):
            for u in range(W):
                X, Y = self.get_ground_coordinates(u, v, W, H, G_w, G_h)
                # Normalize coordinates to image space
                X_img = int(X / G_w * W + W / 2)
                Y_img = int(Y / G_h * H + H / 2)
                if 0 <= X_img < W and 0 <= Y_img < H:
                    warped_image[Y_img, X_img] = image[v, u]
    
        return warped_image

    def get_all_images(self):
        return self.images
     
    def horStitch(self, stitchList, images):
        for i in stitchList:
            stitched_image = None
            for j in i:
                camera_name = f'Camera {j + 1}'
                gray_image = images[camera_name]['gray']
                if i.index(j) == 0:
                    stitched_image = gray_image
                else:
                    stitched_image = np.concatenate((stitched_image, gray_image), axis=1)
            stitched_images.append(stitched_image)

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
    # Extract birdseye view images
    image1 = images['Camera 1']['birdseye']
    image2 = images['Camera 2']['birdseye']
    image3 = images['Camera 3']['birdseye']
    image4 = images['Camera 4']['birdseye']

    # Get dimensions of each image
    h1, w1 = image1.shape
    h2, w2 = image2.shape
    h3, w3 = image3.shape
    h4, w4 = image4.shape

    # Determine dimensions of the stitched image
    max_height = max(h1 + h3, h2)
    max_width = max(w2 + w4, w1)

    # Create an empty canvas for the stitched image
    stitched_image = np.zeros((max_height, max_width), dtype=np.uint8)

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
    # Apply adaptive thresholding
    object_map = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
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

