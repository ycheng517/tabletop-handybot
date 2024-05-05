import json
import os

import cv2
import numpy as np
import open3d as o3d
import openai
import rclpy
import supervision as sv
import torch
import torchvision
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from groundingdino.util.inference import Model
from openai.types.beta import Assistant
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from segment_anything import SamPredictor, sam_model_registry
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String

from pymoveit2 import GripperInterface, MoveIt2

from .openai_assistant import get_or_create_assistant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/yifei/src/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = (
    "/home/yifei/src/Grounded-Segment-Anything/groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/yifei/src/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray,
            xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box,
                                                 multimask_output=True)
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

        self.logger = self.get_logger()

        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.openai = openai.OpenAI()
        self.assistant: Assistant = get_or_create_assistant(self.openai)

        self.cv_bridge = CvBridge()
        self.gripper_joint_name = "gripper_joint"
        callback_group = ReentrantCallbackGroup()
        # Create MoveIt 2 interface
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=[
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                "joint_6"
            ],
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"

        self.gripper_interface = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[-0.012],
            closed_gripper_joint_positions=[0.0],
            gripper_group_name="ar_gripper",
            callback_group=callback_group,
            gripper_command_action_name="/gripper_controller/gripper_cmd",
        )

        self.annotate = annotate
        self.n_frames_processed = 0
        self._last_depth_msg = None
        self._last_rgb_msg = None

        self.image_sub = self.create_subscription(Image,
                                                  "/camera/color/image_raw",
                                                  self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw",
            self.depth_callback, 10)

        self.point_cloud_pub = self.create_publisher(
            PointCloud2, "/grounded_sam/point_cloud", 2)
        self.prompt_sub = self.create_subscription(String, "/prompt",
                                                   self.start, 10)
        self.save_images_sub = self.create_subscription(
            String, "/save_images", self.save_images, 10)
        self.release_at_sub = self.create_subscription(Pose, "/release_at",
                                                       self.release_at, 10)

        self.logger.info("Grounded SAM node initialized.")

    def start(self, msg: String):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)

        self.logger.info(f"Processing: {msg.data}")
        thread = self.openai.beta.threads.create()
        message = self.openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=msg.data,
        )
        self.logger.info(message)

        run = self.openai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        )

        done = False
        while not done:
            if run.status == "completed":
                messages = self.openai.beta.threads.messages.list(
                    thread_id=thread.id)
                self.logger.info(messages)
                done = True
                break
            else:
                self.logger.info(run.status)

            detections = None
            tool_outputs = []
            for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                self.logger.info(f"tool_call: {tool_call}")
                if tool_call.type == "function":
                    if tool_call.function.name == "get_objects":
                        args = json.loads(tool_call.function.arguments)
                        classes = args["object_classes"]
                        detections = self.detect_objects(rgb_image, classes)
                        detected_classes = [
                            classes[class_id]
                            for class_id in detections.class_id
                        ]
                        self.logger.info(f"Detected {detected_classes}.")
                        tool_outputs.append({
                            "type":
                            "object_detection_result",
                            "object_classes":
                            detected_classes,
                            "confidence":
                            detections.confidence.tolist(),
                            "xyxy":
                            detections.xyxy.tolist(),
                        })
                    elif tool_call.function.name == "execute_action":
                        args = json.loads(tool_call.function.arguments)
                        if detections is None:
                            self.logger.error(
                                "No detections available to execute action.")
                            continue

                        result = self.execute_action(
                            args["action"],
                            args["object_index"],
                            detections,
                            depth_image,
                        )
                        tool_outputs.append({
                            "type": "action_result",
                            "success": result,
                        })
            self.logger.info(f"tool_outputs: {tool_outputs}")

            if tool_outputs:
                try:
                    run = self.openai.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs)
                    self.logger.info("Tool outputs submitted successfully.")
                except Exception as e:  # pylint: disable=broad-except
                    self.logger.error(f"Failed to submit tool outputs: {e}")
            else:
                self.logger.info("No tool outputs to submit.")

    def detect_objects(self, image: np.ndarray, object_classes: str):
        detections: sv.Detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=[object_classes],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # NMS post process
        nms_idx = (torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD,
        ).numpy().tolist())

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
                f"{object_classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _ in detections
            ]
            annotated_frame = box_annotator.annotate(scene=image.copy(),
                                                     detections=detections,
                                                     labels=labels)

            annotated_frame = mask_annotator.annotate(scene=image.copy(),
                                                      detections=detections)
            cv2.imwrite(f"annotated_image_{self.n_frames_processed}.jpg",
                        annotated_frame)

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
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        )

        # convert it to a ROS PointCloud2 message
        points = np.asarray(pcd.points)
        pc_msg = point_cloud(points, "/camera_color_frame")
        self.point_cloud_pub.publish(pc_msg)

        self.n_frames_processed += 1
        return detections

    def execute_action(
        self,
        action: str,
        object_index: int,
        detections: sv.Detections,
        depth_image: np.ndarray,
    ) -> bool:
        if action == "pick_object":
            # pick the object top down
            # mask out the depth image except for the detected objects
            masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
            mask = detections.mask[object_index]
            masked_depth_image[mask] = depth_image[mask]
            masked_depth_image /= 1000.0

            # convert the masked depth image to a point cloud
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(masked_depth_image),
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.
                    PrimeSenseDefault),
            )
            # TODO(Yifei): transform point cloud to robot base frame
            points = np.asarray(pcd.points)

            min_z = points[:, 2].min()
            grasp_z = min_z + (points[:, 2].max() - min_z) / 2
            # get minAreaRect of the object in top-down view
            center, dimensions, theta = cv2.minAreaRect(points[:, :2])
            print(center)
            print(dimensions)
            print(theta)

        elif action == "move_above_object_and_release":
            # move the robot arm above the object and release the object
            pass
        return True

    def release_at(self, msg: Pose):
        # NOTE: straight down is wxyz 0, 0, 1, 0
        # good pose is 0, -0.3, 0.35
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = msg

        self.moveit2.move_to_pose(pose=pose_goal)
        self.moveit2.wait_until_executed()

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

        # self.gripper_interface.close()
        # self.gripper_interface.wait_until_executed()

    def depth_callback(self, msg):
        self._last_depth_msg = msg

    def image_callback(self, msg):
        self._last_rgb_msg = msg

    def save_images(self, msg):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)

        save_dir = msg.data
        cv2.imwrite(
            os.path.join(save_dir, f"rgb_image_{self.n_frames_processed}.png"),
            rgb_image)
        np.save(
            os.path.join(save_dir,
                         f"depth_image_{self.n_frames_processed}.npz"),
            depth_image)


def main():
    rclpy.init()
    node = GroundedSAMNode()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
