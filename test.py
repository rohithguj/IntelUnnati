#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageStitcher(Node):
    def __init__(self):
        super().__init__('image_stitcher')
        self.bridge = CvBridge()
        self.image_list = [None] * 4  # List to store captured frames from all cameras
        self.image_counter = [0] * 4  # List to store image counters for each camera

        # Create directory to save images
        self.image_dir = "camera_feeds"
        os.makedirs(self.image_dir, exist_ok=True)

        # Create subscriptions for each camera
        for i in range(4):
            topic = f'overhead_camera/overhead_camera{i + 1}/image_raw'
            self.create_subscription(Image, topic, lambda msg, index=i: self.image_callback(msg, index), 10)
            self.get_logger().info(f'Subscribed to {topic}')

    def image_callback(self, msg, camera_index):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image_list[camera_index] = cv_image

        # Save individual camera feed
        image_filename = os.path.join(self.image_dir, f'camera_{camera_index + 1}_frame_{self.image_counter[camera_index]}.jpg')
        cv2.imwrite(image_filename, cv_image)
        self.image_counter[camera_index] += 1

        # Stitch images when all 4 cameras have contributed frames
        if all(image is not None for image in self.image_list):
            stitched_image = self.stitch_images(self.image_list)
            cv2.imshow("Stitched Image", stitched_image)
            cv2.waitKey(1)  # Adjusted to show the stitched image without blocking

    def stitch_images(self, images):
        # Perform image stitching here (e.g., using OpenCV's stitcher)
        # You can use methods like cv2.Stitcher_create() or manual stitching
        # Return the stitched image
        # For demonstration purposes, let's just concatenate the images side by side
        return cv2.hconcat(images)

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

