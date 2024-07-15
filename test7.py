import cv2
import numpy as np
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class HouseMapStitcher(Node):
    def __init__(self):
        super().__init__('house_map_stitcher')
        self.num_cams = 4
        self.bridge = CvBridge()
        self.images = [None] * self.num_cams
        self.keypoints = [None] * self.num_cams
        self.descriptors = [None] * self.num_cams
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.stitched_image = None

        self.image_subscriptions = []
        self.image_count = [0] * self.num_cams

        self.image_dir = "camera_feeds"
        os.makedirs(self.image_dir, exist_ok=True)

        for i in range(self.num_cams):
            topic = f'/overhead_camera/overhead_camera{i + 1}/image_raw'
            self.image_subscriptions.append(
                self.create_subscription(Image, topic, lambda msg, index=i: self.image_callback(msg, index), 10)
            )
            self.get_logger().info(f'Subscribed to {topic}')

    def image_callback(self, msg, camera_index):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.images[camera_index] = cv_image
        self.image_count[camera_index] += 1

        image_filename = os.path.join(
            self.image_dir, f'camera_{camera_index + 1}_frame_{self.image_count[camera_index]}.jpg'
        )
        cv2.imwrite(image_filename, cv_image)
        self.get_logger().info(f'Received image from camera {camera_index + 1}')

        if all(image is not None for image in self.images):
            self.get_logger().info('All images received, starting stitching process...')
            self.process_images()
            self.stitch_images()
            self.display_stitched_image()

    def process_images(self):
        for i in range(self.num_cams):
            self.detect_and_compute_features(i)

    def detect_and_compute_features(self, index):
        orb = cv2.ORB_create()
        self.keypoints[index], self.descriptors[index] = orb.detectAndCompute(self.images[index], None)

    def stitch_images(self):
        # Match features between pairs of images
        matches = []
        for i in range(self.num_cams - 1):
            matches.append(self.matcher.match(self.descriptors[i], self.descriptors[i+1]))

        # Sort matches by score
        for match in matches:
            match.sort(key=lambda x: x.distance, reverse=False)

        # Extract location of good matches
        if len(matches) > 0 and len(matches[0]) > 0:
            src_pts = np.float32([self.keypoints[i][m.queryIdx].pt for m in matches[0]]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints[i+1][m.trainIdx].pt for m in matches[0]]).reshape(-1, 1, 2)

            # Find homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Use homography to warp images
            h, w = self.images[1].shape[:2]
            self.stitched_image = cv2.warpPerspective(self.images[1], M, (w, h))
        else:
            self.get_logger().warning('No valid matches found for stitching.')

    def display_stitched_image(self):
        if self.stitched_image is not None:
            cv2.imshow('Stitched Image', self.stitched_image)
            cv2.waitKey(1)  # Adjust waitKey value as needed
            # Uncomment below line if running in ROS environment to handle image display properly
            # rclpy.spin_once(self)

def main(args=None):
    rclpy.init(args=args)
    stitcher = HouseMapStitcher()
    rclpy.spin(stitcher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

