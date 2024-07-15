import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node

NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
CAMERA_POSITIONS = [[-5, -2, CAMERA_HEIGHT_IN_METERS], [-5, 3, CAMERA_HEIGHT_IN_METERS], [1, -2, CAMERA_HEIGHT_IN_METERS], [1, 3, CAMERA_HEIGHT_IN_METERS]]

class ImageStitcher(Node):
    def __init__(self):
        super().__init__('image_stitcher')
        self.bridge = CvBridge()
        self.image_list = [None] * NUM_OF_CAMS
        self.image_counter = [0] * NUM_OF_CAMS

        self.image_dir = "camera_feeds"
        os.makedirs(self.image_dir, exist_ok=True)

        # Initialize ORB detector and FLANN matcher
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2)
        self.flann = cv2.FlannBasedMatcher({'algorithm': 6, 'trees': 5}, {'checks': 50})

        # Subscribe to camera topics
        self.image_subscribers = []
        for i in range(NUM_OF_CAMS):
            topic = f'overhead_camera/overhead_camera{i + 1}/image_raw'
            sub = self.create_subscription(Image, topic, lambda msg, index=i: self.image_callback(msg, index), 10)
            self.image_subscribers.append(sub)
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
                # Save stitched image
                stitched_filename = os.path.join(self.image_dir, 'stitched_image.jpg')
                cv2.imwrite(stitched_filename, stitched_image)
                self.show_image(stitched_image, "Stitched Image")
                self.create_map(stitched_image)  # Create 2D map from stitched image
            else:
                self.get_logger().error('Error stitching images: Result is None')

    def custom_stitch_images(self, image_list):
        # Ensure all images are available
        if len(image_list) < NUM_OF_CAMS:
            return None

        # Detect keypoints and compute descriptors for each image
        kp_list = []
        des_list = []
        for img in image_list:
            kp, des = self.orb.detectAndCompute(img, None)
            kp_list.append(kp)
            des_list.append(des if des is not None else np.empty((0, 32), dtype=np.uint8))  # Handle empty descriptors

        # Find base image based on camera position (choose the one closest to origin)
        base_cam = min(range(NUM_OF_CAMS), key=lambda x: np.linalg.norm(np.array(CAMERA_POSITIONS[x][:2])))

        # Compute homography for each camera relative to the base camera based on object features
        homography_matrices = {}
        for cam in range(NUM_OF_CAMS):
            if cam != base_cam and des_list[base_cam].shape[0] > 0 and des_list[cam].shape[0] > 0:
                # Match descriptors between base_cam and cam using FLANN matcher
                matches = self.flann.knnMatch(des_list[base_cam], des_list[cam], k=2)
                
                # Apply ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                # Check if enough good matches were found
                if len(good_matches) >= 4:
                    # Extract matched keypoints
                    src_pts = np.float32([kp_list[base_cam][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_list[cam][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    # Calculate homography using RANSAC
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
                    homography_matrices[(base_cam, cam)] = H
                else:
                    self.get_logger().error(f'Not enough good matches found between camera {base_cam + 1} and {cam + 1}')

        # Example: Use computed homography to stitch images
        stitched_images = []
        for (base_cam, cam), H in homography_matrices.items():
            if base_cam < cam:
                # Warp cam image onto base_cam coordinate system
                result = cv2.warpPerspective(image_list[cam], H, (image_list[base_cam].shape[1] + image_list[cam].shape[1], image_list[base_cam].shape[0]))
                result[0:image_list[base_cam].shape[0], 0:image_list[base_cam].shape[1]] = image_list[base_cam]
                stitched_images.append(result)

        # Combine stitched images into final map (as per your specific layout)
        if len(stitched_images) == 2:  # Assuming 2 stitched pairs
            final_map = np.concatenate(stitched_images, axis=0)
        else:
            final_map = None

        return final_map

    def resize_image(self, image, size):
        return cv2.resize(image, size)

    def show_image(self, image, window_name):
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

    def create_map(self, stitched_image):
        # Process the stitched image to create a 2D map (example: save as an image file)
        map_filename = os.path.join(self.image_dir, 'godown_map.jpg')
        cv2.imwrite(map_filename, stitched_image)
        self.get_logger().info(f'Created 2D map: {map_filename}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

