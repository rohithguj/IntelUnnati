import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

NUM_OF_CAMS = 4

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.images = {}
        
        # Create subscriptions for each camera
        self.subscription1 = self.create_subscription(Image, '/overhead_camera/overhead_camera3/image_raw', self.listener_callback1, 10)
        self.subscription2 = self.create_subscription(Image, '/overhead_camera/overhead_camera2/image_raw', self.listener_callback2, 10)
        self.subscription3 = self.create_subscription(Image, '/overhead_camera/overhead_camera1/image_raw', self.listener_callback3, 10)
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
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Display the grayscale image
        cv2.imshow(f'{camera_name} - Grayscale', cv_image_gray)
        cv2.waitKey(1)

        # Detect and extract the top parts of objects
        _, thresh = cv2.threshold(cv_image_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        top_parts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            top_part = cv_image[y:y+int(h/2), x:x+w]  # Take the top half of the bounding box
            top_parts.append((top_part, (x, y, w, int(h/2))))

            # Display the extracted top part
            cv2.imshow(f'{camera_name} - Top Part', top_part)
            cv2.waitKey(1)

        self.images[camera_name] = top_parts

    def create_2d_map(self):
        # Initialize an empty canvas for the 2D map
        map_width = 2000  # Example width of the map
        map_height = 2000 # Example height of the map
        map_canvas = np.zeros((map_height, map_width, 3), dtype=np.uint8)

        # Example coordinate offsets (should be adapted based on actual camera setup)
        x_offset = 0
        y_offset = 0

        for camera_name, parts in self.images.items():
            for part, (x, y, w, h) in parts:
                # Define the destination position for each part on the map
                dest_x = x + x_offset
                dest_y = y + y_offset

                # Check if part fits within map boundaries
                if dest_x + w <= map_width and dest_y + h <= map_height:
                    map_canvas[dest_y:dest_y+h, dest_x:dest_x+w] = part

                # Display the current state of the 2D map
                cv2.imshow('2D Map', map_canvas)
                cv2.waitKey(1)

        return map_canvas

    def save_results(self, image):
        # Save as PGM file
        pgm_filename = 'result.pgm'
        cv2.imwrite(pgm_filename, image)
        self.get_logger().info(f'Saved 2D map as {pgm_filename}')

        # Save as YAML file (metadata)
        yaml_filename = 'result.yaml'
        with open(yaml_filename, 'w') as yaml_file:
            yaml_content = "image: result.pgm\nresolution: 0.05\norigin: [0.0, 0.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196"
            yaml_file.write(yaml_content)
        self.get_logger().info(f'Saved metadata as {yaml_filename}')

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            
            # Process images and create the 2D map
            if len(node.images) == NUM_OF_CAMS:
                map_canvas = node.create_2d_map()
                node.save_results(map_canvas)
                
                # Display final 2D map
                cv2.imshow("Final 2D Map", map_canvas)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    


