import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node

NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
CAMERA_POSITIONS = [
    [-5, -2, CAMERA_HEIGHT_IN_METERS], 
    [-5, 3, CAMERA_HEIGHT_IN_METERS], 
    [1, -2, CAMERA_HEIGHT_IN_METERS], 
    [1, 3, CAMERA_HEIGHT_IN_METERS]
]

class ImageStitcher(Node):

    def __init__(self):
        super().__init__('image_stitcher')
        self.bridge = CvBridge()
        self.image_list = [None] * NUM_OF_CAMS
        self.image_counter = [0] * NUM_OF_CAMS

        self.image_dir = "camera_feeds"
        os.makedirs(self.image_dir, exist_ok=True)

        for i in range(NUM_OF_CAMS):
            topic = f'overhead_camera/overhead_camera{i + 1}/image_raw'
            self.create_subscription(Image, topic, lambda msg, index=i: self.image_callback(msg, index), 10)
            self.get_logger().info(f'Subscribed to {topic}')

    def image_callback(self, msg, camera_index):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image_list[camera_index] = cv_image

        image_filename = os.path.join(self.image_dir, f'camera_{camera_index + 1}_frame_{self.image_counter[camera_index]}.jpg')
        cv2.imwrite(image_filename, cv_image)
        self.image_counter[camera_index] += 1

        self.get_logger().info(f'Received image from camera {camera_index + 1}')

        if all(image is not None for image in self.image_list):
            self.get_logger().info('All images received, starting stitching process...')
            stitched_image = self.custom_stitch_images(self.image_list)
            if stitched_image is not None:
                resized_image = self.resize_image(stitched_image, (520, 520))
                stitched_filename = os.path.join(self.image_dir, 'stitched_image.jpg')
                cv2.imwrite(stitched_filename, resized_image)
                pgm_filename = os.path.join(self.image_dir, 'stitched_image.pgm')
                yaml_filename = os.path.join(self.image_dir, 'stitched_image.yaml')
                self.save_pgm_and_yaml(resized_image, pgm_filename, yaml_filename)
                cv2.imshow("Stitched Image", resized_image)
                cv2.waitKey(1)
            else:
                self.get_logger().error('Error stitching images: Result is None')

    def custom_stitch_images(self, image_list):
        stitched_image = np.zeros((520, 520, 3), dtype=np.uint8)
        
        # Bird's eye view coordinates
        bird_eye_view = np.array([
            [0, 0],
            [0, 520],
            [520, 0],
            [520, 520]
        ], dtype=np.float32)
        
        # Loop through each camera image and its position
        for i in range(len(image_list)):
            image = image_list[i]
            
            # Skip if image is None
            if image is None:
                continue
            
            # Define points for perspective transformation
            src_pts = np.float32([
                [0, 0],
                [0, image.shape[0]],
                [image.shape[1], 0],
                [image.shape[1], image.shape[0]]
            ])
            
            # Calculate perspective transformation matrix (homography)
            H, _ = cv2.findHomography(src_pts, bird_eye_view, cv2.RANSAC, 5.0)
            
            # Warp the current camera image to bird's eye view
            warped_image = cv2.warpPerspective(image, H, (520, 520))
            
            # Add the warped image to the stitched image
            stitched_image += warped_image
        
        return stitched_image

    def resize_image(self, image, dimensions):
        return cv2.resize(image, dimensions)

    def save_pgm_and_yaml(self, image, pgm_filename, yaml_filename):
        cv2.imwrite(pgm_filename, image)
        # Optionally save other metadata in YAML format
        # Example:
        # with open(yaml_filename, 'w') as f:
        #     yaml.dump(metadata_dict, f)

def main(args=None):
    rclpy.init(args=args)
    image_stitcher = ImageStitcher()
    rclpy.spin(image_stitcher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

