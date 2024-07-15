import os
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
        
        # Save original image
        original_image_filename = os.path.join(self.image_dir, f'camera_{camera_index + 1}_frame_{self.image_counter[camera_index]}.jpg')
        cv2.imwrite(original_image_filename, cv_image)

        self.image_list[camera_index] = cv_image
        self.image_counter[camera_index] += 1

        self.get_logger().info(f'Received image from camera {camera_index + 1}')

        if all(image is not None for image in self.image_list):
            self.get_logger().info('All images received, starting stitching process...')
            stitched_image = self.custom_stitch_images(self.image_list)
            if stitched_image is not None:
                resized_image = self.resize_image(stitched_image, (800, 800))
                stitched_filename = os.path.join(self.image_dir, 'stitched_image.jpg')
                cv2.imwrite(stitched_filename, resized_image)
                cv2.imshow("Stitched Image", resized_image)
                cv2.waitKey(1)
            else:
                self.get_logger().error('Error stitching images: Result is None')

    def custom_stitch_images(self, image_list):
        num_cams = len(image_list)
        if num_cams < 2:
            return None
        
        # Assuming you have homography matrices calculated for each pair of cameras
        # Adjust the following for your actual homography matrices
        homography_matrices = [
            np.eye(3),  # Identity matrix for the first camera
            self.calculate_homography_matrix(CAMERA_POSITIONS[0], CAMERA_POSITIONS[1]),
            self.calculate_homography_matrix(CAMERA_POSITIONS[0], CAMERA_POSITIONS[2]),
            self.calculate_homography_matrix(CAMERA_POSITIONS[0], CAMERA_POSITIONS[3])
        ]
        
        # Initialize stitched image size (adjust based on your requirement)
        stitched_width = 800
        stitched_height = 800
        stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
        
        # Warp and concatenate images
        for i in range(num_cams):
            img = image_list[i]
            H = homography_matrices[i]
            
            # Warp image
            warped_img = cv2.warpPerspective(img, H, (stitched_width, stitched_height))
            
            # Concatenate warped image onto stitched image
            if i == 0:
                stitched_image = warped_img
            else:
                # Ensure the overlap area is properly aligned and blended
                stitched_image = self.blend_images(stitched_image, warped_img)
        
        return stitched_image

    def blend_images(self, img1, img2):
        # Example blending function (you may adjust as needed)
        return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    def calculate_homography_matrix(self, pos1, pos2):
        # Calculate homography matrix assuming the ground plane is flat
        src_pts = np.float32([[pos1[0], pos1[1]],
                              [pos1[0], pos1[1] + 1],
                              [pos1[0] + 1, pos1[1]],
                              [pos1[0] + 1, pos1[1] + 1]])

        dst_pts = np.float32([[pos2[0], pos2[1]],
                              [pos2[0], pos2[1] + 1],
                              [pos2[0] + 1, pos2[1]],
                              [pos2[0] + 1, pos2[1] + 1]])

        return cv2.findHomography(src_pts, dst_pts)[0]

    def resize_image(self, image, size):
        return cv2.resize(image, size)

def main(args=None):
    rclpy.init(args=args)
    image_stitcher = ImageStitcher()
    rclpy.spin(image_stitcher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

