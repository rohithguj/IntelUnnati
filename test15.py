import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Constants
NUM_OF_CAMS = 4
CAMERA_HEIGHT_IN_METERS = 8
FOCAL_LENGTH = 1.0
CAMERA_POSITIONS = [
    [-5, -2, CAMERA_HEIGHT_IN_METERS],
    [-5, 3, CAMERA_HEIGHT_IN_METERS],
    [1, -2, CAMERA_HEIGHT_IN_METERS],
    [1, 3, CAMERA_HEIGHT_IN_METERS]
]

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

class ImageStitcher:
    def __init__(self):
        pass

    def stitch_images_locally(self, image1, image2, overlap_percent=0.4, direction='horizontal'):
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

        self.verList = []
        self.populate_verList()

    def populate_verList(self):
        # Populate verList and sort based on element[1] at the same time
        self.verList = sorted(
            (
                sorted(
                    [index for index, element in enumerate(CAMERA_POSITIONS) if element[0] == x],
                    key=lambda idx: -CAMERA_POSITIONS[idx][1]  # Sort by y coordinate in descending order
                )
                for x in sorted(set(element[0] for element in CAMERA_POSITIONS), reverse=True)  # Sort x values in reverse order
            ),
            key=lambda sublist: -CAMERA_POSITIONS[sublist[0]][0]  # Sort by x coordinate in descending order
        )

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

        # Perform object detection using YOLOv3
        color_image, depth_image = self.detect_objects(cv_image)

        # Store both color and depth images
        self.images[camera_name] = {
            'color': color_image,
            'depth': depth_image,
            'original': cv_image  # Store original color image for visualization, if needed
        }

    def detect_objects(self, color_image, confidence_threshold=0.5, nms_threshold=0.4):
        # Perform object detection
        height, width, _ = color_image.shape
        blob = cv2.dnn.blobFromImage(color_image, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Extract bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    # Scale back to original size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        # Extract top parts of detected objects
        top_parts = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            # Extracting top 30% of the bounding box height
            top_part_bbox = [x, y, x + w, y + int(0.3 * h)]  # Adjust the percentage as needed
            top_part_image = color_image[top_part_bbox[1]:top_part_bbox[3], top_part_bbox[0]:top_part_bbox[2]]
            top_parts.append(top_part_image)

        return color_image, top_parts

    def vertStitch(self):
        stitched_images = []

        for i in self.verList:
            stitched_row = None
            for j in i:
                camera_name = f'Camera {j + 1}'
                color_image = self.images[camera_name]['color']
                top_parts = self.images[camera_name]['depth']  # Assuming depth is used for stitching

                if stitched_row is None:
                    stitched_row = top_parts[0]  # Take the first top part
                else:
                    # Stitch with overlap using ImageStitcher
                    stitched_row = self.stitcher.stitch_images_locally(stitched_row, top_parts[0], direction='horizontal')

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

def main(args=None):
    rclpy.init(args=args)
    node = ImageListener()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

            # Check if we have all images
            if len(node.images) == NUM_OF_CAMS:
                # Perform vertical stitching
                vertical_stitched = node.vertStitch()

                # Perform horizontal stitching
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

