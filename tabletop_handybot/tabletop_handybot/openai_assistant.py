from openai import OpenAI  # pylint: disable=import-self
from openai.types.beta import Assistant

# pylint: disable=line-too-long


def get_or_create_assistant(client: OpenAI,
                            assistant_id: str = "") -> Assistant:
    if assistant_id:
        return client.beta.assistants.retrieve(assistant_id)

    return client.beta.assistants.create(
        name="Tabletop Assistant",
        instructions=(
            "You are a robot arm mounted on a table. Write and run code to "
            "do tasks on the table. You can only pick up one object at a time."
        ),
        model="gpt-4o",
        temperature=0.01,
        tools=[{
            "type": "function",
            "function": {
                "name": "detect_objects",
                "description":
                "Detect objects in the field of view of the camera",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_classes": {
                            "type":
                            "string",
                            "description":
                            ("Object classes to detect, comma separated"
                             "For example: horses,rivers,plain"),
                        }
                    },
                    "required": ["object_classes"],
                },
            },
        }, {
            "type": "function",
            "function": {
                "name": "pick_object",
                "description":
                "Pick up an object from the output of get_objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_index": {
                            "type":
                            "integer",
                            "description":
                            "index of target object in the detected objects list to execute the action for."
                        }
                    },
                    "required": ["object_index"]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "move_above_object_and_release",
                "description":
                "move the end effector above the object and release the gripper. The object is from the output of get_objects",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_index": {
                            "type":
                            "integer",
                            "description":
                            "index of target object in the detected objects list to execute the action for."
                        }
                    },
                    "required": ["object_index"]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "release_gripper",
                "description": "Open up the gripper",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "flick_wrist_while_release",
                "description":
                "Flick the wrist while releasing the gripper, basically tossing the object.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }],
    )
