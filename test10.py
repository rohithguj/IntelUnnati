import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Assuming you have some constant like CAMERA_POSITIONS defined
CAMERA_POSITIONS = [
    [0, 0],
    [0, 100],
    [100, 0],
    [100, 100]
]
NUM_OF_CAMS = 4

class ImageListener(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.bridge = CvBridge()
        self.images = {}  # Dictionary to store images
        self.objects = {}  # Dictionary to store detected objects

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

        # Perform blob detection (example using SimpleBlobDetector)
        keypoints = self.detect_blobs(cv_image)

        # Store detected objects and images
        self.objects[camera_name] = keypoints

        # Warp image to top-down view
        warped_image = self.warp_image_to_top_down(cv_image)
        self.images[camera_name] = warped_image

    def detect_blobs(self, image):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 100
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        return keypoints

    def warp_image_to_top_down(self, camera_image):
        # Example parameters
        H = 5  # Camera height in meters
        FOV = 90  # Field of view in degrees
        Rx, Ry = camera_image.shape[1], camera_image.shape[0]  # Camera resolution in pixels (horizontal, vertical)

        # Assuming intrinsic matrix K is known
        K = np.array([[1000, 0, Rx/2],
                      [0, 1000, Ry/2],
                      [0, 0, 1]])

        # Compute transformation matrix T
        focal_length = Rx / (2 * np.tan(np.radians(FOV / 2)))
        T = np.array([[focal_length, 0, Rx/2],
                      [0, focal_length, Ry/2],
                      [0, 0, 1]])

        # Initialize top-down view image
        top_down_view = np.zeros((Ry, Rx, 3), dtype=np.uint8)

        # Generate top-down view
        for y in range(Ry):
            for x in range(Rx):
                pixel_camera = np.array([x, y, 1])
                pixel_top_down = np.dot(T, pixel_camera)
                pixel_top_down = pixel_top_down.astype(int)

                if 0 <= pixel_top_down[0] < Rx and 0 <= pixel_top_down[1] < Ry:
                    top_down_view[y, x] = camera_image[pixel_top_down[1], pixel_top_down[0]]

        return top_down_view

    def stitch_images_1_2(self):
        # Extract images from self.images dictionary
        image1 = self.images.get('Camera 1', None)
        image2 = self.images.get('Camera 2', None)

        if image1 is None or image2 is None:
            self.get_logger().warn("Images for Camera 1 or Camera 2 not found.")
            return None

        # Detect blobs in images
        keypoints1 = self.objects.get('Camera 1', [])
        keypoints2 = self.objects.get('Camera 2', [])

        # Warp images to top-down view
        warped_image1 = self.warp_image_to_top_down(image1)
        warped_image2 = self.warp_image_to_top_down(image2)

        # Determine dimensions of the stitched image for cameras 1 and 2
        h1, w1, _ = warped_image1.shape
        h2, w2, _ = warped_image2.shape
        max_height = max(h1, h2)
        max_width = w1 + w2

        # Create an empty canvas for the stitched image
        stitched_image_1_2 = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        # Place each image onto the stitched image canvas for cameras 1 and 2
        stitched_image_1_2[0:h1, 0:w1] = warped_image1
        stitched_image_1_2[0:h2, w1:w1+w2] = warped_image2

        # Draw keypoints on stitched image
        for kp in keypoints1:
            cv2.drawMarker(stitched_image_1_2, (int(kp.pt[0]), int(kp.pt[1])), (255, 0, 0), cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

        for kp in keypoints2:
            kp.pt = (kp.pt[0] + w1, kp.pt[1])
            cv2.drawMarker(stitched_image_1_2, (int(kp.pt[0]), int(kp.pt[1])), (255, 0, 0), cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

        return stitched_image_1_2

    def stitch_images_3_4(self):
        # Extract images from self.images dictionary
        image3 = self.images.get('Camera 3', None)
        image4 = self.images.get('Camera 4', None)

        if image3 is None or image4 is None:
            self.get_logger().warn("Images for Camera 3 or Camera 4 not found.")
            return None

        # Detect blobs in images
        keypoints3 = self.objects.get('Camera 3', [])
        keypoints4 = self.objects.get('Camera 4', [])

        # Warp images to top-down view
        warped_image3 = self.warp_image_to_top_down(image3)
        warped_image4 = self.warp_image_to_top_down(image4)

        # Determine dimensions of the stitched image for cameras 3 and 4
        h3, w3, _ = warped_image3.shape
        h4, w4, _ = warped_image4.shape
        max_height = max(h3, h4)
        max_width = w3 + w4

        # Create an empty canvas for the stitched image
        stitched_image_3_4 = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        # Place each image onto the stitched image canvas for cameras 3 and 4
        stitched_image_3_4[0:h3, 0:w3] = warped_image3
        stitched_image_3_4[0:h4, w3:w3+w4] = warped_image4

        # Draw keypoints on stitched image
        for kp in keypoints3:
            cv2.drawMarker(stitched_image_3_4, (int(kp.pt[0]), int(kp.pt[1])), (255, 0, 0), cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

        for kp in keypoints4:
            kp.pt = (kp.pt[0] + w3, kp.pt[1])
            cv2.drawMarker(stitched_image_3_4, (int(kp.pt[0]), int(kp.pt[1])), (255, 0, 0), cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

        return stitched_image_3_4

    def stitch_combined_images(self, stitched_image_1_2, stitched_image_3_4):
        # Get dimensions of each stitched image
        h1_2, w1_2, _ = stitched_image_1_2.shape
        h3_4, w3_4, _ = stitched_image_3_4.shape

        # Determine dimensions of the final stitched image
        max_height = h1_2 + h3_4
        max_width = max(w1_2, w3_4)

        # Create an empty canvas for the final stitched image
        stitched_combined_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        # Place stitched image of cameras 3 and 4 above stitched image of cameras 1 and 2
        stitched_combined_image[0:h3_4, 0:w3_4] = stitched_image_3_4
        stitched_combined_image[h3_4:h1_2+h3_4, 0:w1_2] = stitched_image_1_2

        return stitched_combined_image


def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

            # Get objects from all cameras
            objects = node.objects

            # Check if we have objects detected from all cameras
            if 'Camera 1' in objects and 'Camera 2' in objects and 'Camera 3' in objects and 'Camera 4' in objects:
                # Stitch images from cameras 1 and 2
                stitched_image_1_2 = node.stitch_images_1_2()

                # Stitch images from cameras 3 and 4
                stitched_image_3_4 = node.stitch_images_3_4()

                if stitched_image_1_2 is not None and stitched_image_3_4 is not None:
                    # Stitch combined images of cameras 1, 2 and 3, 4
                    stitched_combined_image = node.stitch_combined_images(stitched_image_1_2, stitched_image_3_4)

                    # Display or save the final stitched image
                    #cv2.imshow("Stitched Image", stitched_combined_image)
                    #cv2.waitKey(1)
                    
                    output_filename = 'stitched_image.png'
                    cv2.imwrite(output_filename, stitched_combined_image)
                    print(f"Stitched image saved as {output_filename}")

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

