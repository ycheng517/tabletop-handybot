import json

import cv2
import numpy as np
import open3d as o3d
import openai
import rclpy
import supervision as sv
import torch
import torchvision
from cv_bridge import CvBridge
from groundingdino.util.inference import Model
from openai.types.beta import Assistant
from rclpy.node import Node
from segment_anything import SamPredictor, sam_model_registry
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String

from .openai_assistant import get_or_create_assistant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/yifei/src/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = (
    "/home/yifei/src/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
)

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/yifei/src/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Predict classes and hyper-param for GroundingDINO
CLASSES = ["toy . plate"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Prompting SAM with detected boxes
def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def point_cloud(points, parent_frame):
    """Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message

    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0

    References:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointCloud2.html
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/PointField.html
        http://docs.ros.org/melodic/api/std_msgs/html/msg/Header.html

    """
    # In a PointCloud2 message, the point cloud is stored as an byte
    # array. In order to unpack it, we also include some parameters
    # which desribes the size of each individual point.
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()

    # The fields specify what the bytes represents. The first 4 bytes
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [
        PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate("xyz")
    ]

    # The PointCloud2 message also has a header which specifies which
    # coordinate frame it is represented in.
    header = Header(frame_id=parent_frame)

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 3),  # Every point consists of three float32s.
        row_step=(itemsize * 3 * points.shape[0]),
        data=data,
    )


class GroundedSAMNode(Node):

    def __init__(self, annotate: bool = False):
        super().__init__("grounded_sam_node")

        self._logger = self.get_logger()

        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH
        )
        self.sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.openai = openai.OpenAI()
        self.assistant: Assistant = get_or_create_assistant(self.openai)
        self.cv_bridge = CvBridge()

        self.annotate = annotate
        self.n_frames_processed = 0
        self._last_depth_msg = None
        self._last_rgb_msg = None

        self.image_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self.depth_callback, 10
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2, "/grounded_sam/point_cloud", 2
        )
        self.prompt_sub = self.create_subscription(String, "/prompt", self.start, 10)

    def start(self, msg: String):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        self._logger.info(f"Processing: {msg.data}")
        thread = self.openai.beta.threads.create()
        message = self.openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=msg.data,
        )
        self._logger.info(message)

        run = self.openai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        )

        done = False
        while not done:
            if run.status == "completed":
                messages = self.openai.beta.threads.messages.list(thread_id=thread.id)
                self._logger.info(messages)
                done = True
                break
            else:
                self._logger.info(run.status)

            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                self._logger.info(f"tool_call: {tool_call}")
                if tool_call.type == "function":
                    if tool_call.function.name == "get_objects":
                        args = json.loads(tool_call.function.arguments)
                        detections = self.detect_objects(args["object_classes"])
                        detected_classes = [
                            CLASSES[class_id] for class_id in detections.class_id
                        ]
                        self._logger.info(f"Detected {detected_classes}.")
                        tool_outputs.append(
                            {
                                "type": "object_detection",
                                "object_classes": detected_classes,
                                "confidence": detections.confidence.tolist(),
                                "xyxy": detections.xyxy.tolist(),
                            }
                        )
                    elif tool_call.function.name == "execute_action":
                        args = json.loads(tool_call.function.arguments)
                        self._logger.info(f"Executing action: {args['action']}")
                        tool_outputs.append(
                            {
                                "type": "action",
                                "success": True,
                            }
                        )
            self._logger.info(f"tool_outputs: {tool_outputs}")

            if tool_outputs:
                try:
                    run = self.openai.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                    )
                    self._logger.info("Tool outputs submitted successfully.")
                except Exception as e:  # pylint: disable=broad-except
                    self._logger.error("Failed to submit tool outputs:", e)
            else:
                self._logger.info("No tool outputs to submit.")

    def detect_objects(self, object_classes: str):
        image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)

        detections: sv.Detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=[object_classes],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        if self.annotate:
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _ in detections
            ]
            annotated_frame = box_annotator.annotate(
                scene=image.copy(), detections=detections, labels=labels
            )

            annotated_frame = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            cv2.imwrite(
                f"annotated_image_{self.n_frames_processed}.jpg", annotated_frame
            )

        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        # mask out the depth image except for the detected objects
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        for mask in detections.mask:
            masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ),
        )

        # convert it to a ROS PointCloud2 message
        points = np.asarray(pcd.points)
        pc_msg = point_cloud(points, "/camera_color_frame")
        self.point_cloud_pub.publish(pc_msg)

        self.n_frames_processed += 1
        return detections

    def depth_callback(self, msg):
        self._last_depth_msg = msg

    def image_callback(self, msg):
        self._last_rgb_msg = msg


def main():
    rclpy.init()
    node = GroundedSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
