#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

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
                pgm_filename = os.path.join(self.image_dir, 'stitched_image.pgm')
                yaml_filename = os.path.join(self.image_dir, 'stitched_image.yaml')
                self.save_pgm_and_yaml(resized_image, pgm_filename, yaml_filename)
                cv2.imshow("Stitched Image", resized_image)
                cv2.waitKey(1)
            else:
                self.get_logger().error('Error stitching images: Result is None')

    def custom_stitch_images(self, images):
        try:
            # Convert images to grayscale
            gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

            # Detect ORB keypoints and descriptors
            orb = cv2.ORB_create()
            self.get_logger().info('ORB detector created')
            keypoints_and_descriptors = [orb.detectAndCompute(image, None) for image in gray_images]

            for idx, (kp, des) in enumerate(keypoints_and_descriptors):
                self.get_logger().info(f'Camera {idx + 1}: {len(kp)} keypoints detected')

            # Initialize matcher
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.get_logger().info('BFMatcher created')

            # Match descriptors between images
            matches = []
            for i in range(NUM_OF_CAMS - 1):
                matches.append(matcher.match(keypoints_and_descriptors[i][1], keypoints_and_descriptors[i + 1][1]))
                self.get_logger().info(f'Found {len(matches[-1])} matches between image {i + 1} and image {i + 2}')

            # Estimate initial homographies based on known camera positions
            homographies = [np.eye(3)]
            for i in range(1, NUM_OF_CAMS):
                H = self.estimate_homography(CAMERA_POSITIONS[i - 1], CAMERA_POSITIONS[i])
                homographies.append(H)
                self.get_logger().info(f'Initial homography for camera {i + 1}: {H}')

            # Refine homographies using feature matches
            for i in range(len(matches)):
                src_pts = np.float32([keypoints_and_descriptors[i][0][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints_and_descriptors[i + 1][0][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                homographies[i + 1] = np.dot(homographies[i], H)
                self.get_logger().info(f'Refined homography for camera {i + 2}: {homographies[i + 1]}')

            # Warp images to a common plane
            base_image = images[0]
            for i in range(1, NUM_OF_CAMS):
                warped_image = cv2.warpPerspective(images[i], homographies[i], (base_image.shape[1] + images[i].shape[1], base_image.shape[0]))
                base_image = self.blend_images(base_image, warped_image)

            return base_image
        except Exception as e:
            self.get_logger().error(f'Error in custom_stitch_images: {e}')
            return None

    def estimate_homography(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]

        # Assuming a simple translation model for initial guess
        H = np.array([[1, 0, dx], 
                      [0, 1, dy], 
                      [0, 0, 1]])

        return H

    def blend_images(self, base_image, warped_image):
        # Create a mask for the base image
        base_mask = np.zeros_like(base_image, dtype=np.uint8)
        base_mask[:, :base_image.shape[1]] = base_image

        # Create a mask for the warped image
        warped_mask = np.zeros_like(warped_image, dtype=np.uint8)
        warped_mask[:, :warped_image.shape[1]] = warped_image

        # Blend images using feathering
        blended_image = cv2.addWeighted(base_mask, 0.5, warped_mask, 0.5, 0)

        return blended_image

    def resize_image(self, image, dimensions):
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

    def save_pgm_and_yaml(self, image, pgm_filename, yaml_filename):
        # Convert to grayscale and save as PGM
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(pgm_filename, grayscale_image)

        # Create a basic YAML file with the image metadata
        height, width = grayscale_image.shape
        yaml_content = f"""image: {pgm_filename}
resolution: 0.05  # Example resolution, adjust as needed
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(yaml_filename, 'w') as yaml_file:
            yaml_file.write(yaml_content)

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

