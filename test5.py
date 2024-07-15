import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

NUM_OF_CAMS = 4

class ImageStitcher(Node):
    def __init__(self):
        super().__init__('image_stitcher')
        self.bridge = CvBridge()
        self.image_list = [None] * NUM_OF_CAMS
        self.image_counter = [0] * NUM_OF_CAMS

        self.image_dir = "camera_feeds"
        os.makedirs(self.image_dir, exist_ok=True)

        # Initialize ORB detector and BFMatcher
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Subscribe to camera topics
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
                resized_image = self.resize_image(stitched_image, (800, 600))  # Adjust size as needed
                stitched_filename = os.path.join(self.image_dir, 'stitched_image.jpg')
                cv2.imwrite(stitched_filename, resized_image)
                self.show_image(resized_image, "Stitched Image")
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
            des_list.append(des)

        # Example: Match keypoints between camera 1 and camera 2
        matches1_2 = self.bf.match(des_list[0], des_list[1])
        src_pts = np.float32([kp_list[0][m.queryIdx].pt for m in matches1_2]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_list[1][m.trainIdx].pt for m in matches1_2]).reshape(-1, 1, 2)
        H1_2, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Example: Match keypoints between camera 3 and camera 4
        matches3_4 = self.bf.match(des_list[2], des_list[3])
        src_pts = np.float32([kp_list[2][m.queryIdx].pt for m in matches3_4]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_list[3][m.trainIdx].pt for m in matches3_4]).reshape(-1, 1, 2)
        H3_4, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Example: Warp and stitch images
        result1_2 = cv2.warpPerspective(image_list[0], H1_2, (image_list[0].shape[1] + image_list[1].shape[1], image_list[1].shape[0]))
        result1_2[0:image_list[1].shape[0], 0:image_list[1].shape[1]] = image_list[1]

        result3_4 = cv2.warpPerspective(image_list[2], H3_4, (image_list[2].shape[1] + image_list[3].shape[1], image_list[3].shape[0]))
        result3_4[0:image_list[3].shape[0], 0:image_list[3].shape[1]] = image_list[3]

        # Combine stitched images into final map
        final_map = np.concatenate((result1_2, result3_4), axis=0)

        return final_map

    def resize_image(self, image, size):
        return cv2.resize(image, size)

    def show_image(self, image, window_name):
        cv2.imshow(window_name, image)
        cv2.waitKey(1)

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


