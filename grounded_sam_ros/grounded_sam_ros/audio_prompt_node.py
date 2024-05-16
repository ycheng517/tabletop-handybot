import rclpy
import rclpy.node
from std_msgs.msg import Empty, String
from whisper_mic import WhisperMic


class AudioPromptNode(rclpy.node.Node):
    """ROS2 Node that listens for a /listen message and publishes the
    result of the audio prompt to /prompt."""

    def __init__(self):
        super().__init__("audio_prompt_node")

        self.whisper_mic = WhisperMic()

        self.prompt_pub = self.create_publisher(String, "/prompt", 10)
        self.listen_sub = self.create_subscription(Empty, "/listen",
                                                   self.listen, 10)
        self._logger.info("Audio Prompt Node Initialized")

    def listen(self, _: Empty):
        result = self.whisper_mic.listen(timeout=10.0, phrase_time_limit=10.0)
        self._logger.info(f"Prompt: {result}")
        self.prompt_pub.publish(String(data=result))


def main():
    rclpy.init()
    node = AudioPromptNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
