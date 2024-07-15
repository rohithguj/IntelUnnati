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
    [-5, 3, CAMERA_HEIGHT_IN_METERS],
    [1, -2, CAMERA_HEIGHT_IN_METERS],
    [1, 3, CAMERA_HEIGHT_IN_METERS]
]

verList = []
verVisit = []
stitched_images = []

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

# Define points in the camera view (image plane)
points_img = np.array([
    [122, 300],
    [122, 359],
    [459, 300],
    [459, 350],
], dtype=np.float32)

# Define corresponding points on the ground plane (2D map)
points_ground = np.array([
    [0, 0],
    [10, 0],
    [0, 10],
    [10, 10],
], dtype=np.float32)

# Compute homography matrix
H, _ = cv2.findHomography(points_img, points_ground, cv2.RANSAC)

class ImageStitcher:
    def __init__(self):
        pass

    def stitch_images_locally(self, image1, image2, overlap_percent=0.535, direction='horizontal'):
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

        # Warp the image using the computed homography matrix
        if H is not None:
            warped_image = cv2.warpPerspective(cv_image, H, (520, 520))  # Adjust output size as needed
        else:
            warped_image = cv_image

        # Convert to grayscale
        cv_image_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        
        # Store both color and grayscale images
        self.images[camera_name] = {
            'color': cv_image,
            'gray': cv_image_gray,
            'warped': warped_image,
            'original': cv_image  # Store original color image for visualization, if needed
        }

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
                # Display transformed images
                for camera_name, img_data in images.items():
                    cv2.imshow(f'{camera_name} Transformed', img_data['warped'])
                
                cv2.waitKey(1)

                # Perform horizontal stitching
                vertical_stitched = node.vertStitch(verList)

                # Perform vertical stitching
                big_image = node.horStitch(vertical_stitched)
                
                # Create object map from big_image
                object_map = create_object_map(big_image)
                
                # Save object map as PGM and YAML files
                save_object_map(object_map, 'object_map')
                
                # Display stitched image
                cv2.imshow("Stitched Image", big_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

def create_object_map(image):
    # Convert image to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    object_map = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Invert the object map (flip color coding)
    object_map = cv2.bitwise_not(object_map)
    
    return object_map

def save_object_map(object_map, base_filename):
    # Save as PGM file
    pgm_filename = base_filename + '.pgm'
    cv2.imshow("object map", object_map)
    cv2.imwrite(pgm_filename, object_map)

    # Save as YAML file (metadata)
    yaml_filename = base_filename + '.yaml'
    with open(yaml_filename, 'w') as yaml_file:
        yaml_content = "image: {}.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196".format(base_filename)
        yaml_file.write(yaml_content)

if __name__ == '__main__':
    main()

