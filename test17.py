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

class ImageStitcher:
    def __init__(self):
        pass

    def stitch_images_locally(self, image1, image2, overlap_percent=0.4, direction='horizontal'):
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
        
        # Convert image to grayscale
        #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Normalize the grayscale image
        #normalized_gray = (gray_image - np.mean(gray_image)) / np.std(gray_image)
        
        # Convert normalized grayscale image to uint8 for blob detection
        #normalized_gray_uint8 = np.uint8((normalized_gray - normalized_gray.min()) / (normalized_gray.max() - normalized_gray.min()) * 255)

        # Detect blobs using SimpleBlobDetector
        blobs = self.detect_blobs(cv_image)
        
        # Create a blank image to draw the 2D map representation
        map_image = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        
        #cv2.imshow("ushn",cv_image)
        cv2.imshow("test",map_image)
        
        # Draw blobs on the map image
        for blob in blobs:
            x, y = int(blob.pt[0]), int(blob.pt[1])
            size = int(blob.size)
            cv2.circle(map_image, (x, y), size, (255), thickness=2)
        
        # Store both color and grayscale images
        self.images[camera_name] = {
            'color': cv_image,
            'gray': map_image,  # Store 2D map representation
            'original': cv_image  # Store original color image for visualization, if needed
        }

    def detect_blobs(self, image):
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 1000
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(image)
        
clear        return keypoints
         
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
                # Perform horizontal stitching
                vertical_stitched = node.vertStitch(verList)

                # Perform vertical stitching
                big_image = node.horStitch(vertical_stitched)
                
                # Display stitched image
                cv2.imshow("Stitched Image", big_image)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

