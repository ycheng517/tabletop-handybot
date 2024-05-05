from openai import OpenAI  # pylint: disable=import-self
from openai.types.beta import Assistant


ASSISTANT_ID = "asst_WLqCkCKIerFYoaSCEvCmIJdR"


def get_or_create_assistant(client: OpenAI) -> Assistant:
    if ASSISTANT_ID:
        return client.beta.assistants.retrieve(ASSISTANT_ID)

    return client.beta.assistants.create(
        name="Tabletop Assistant",
        instructions=(
            "You are a robot arm mounted on a table. Write and run code to "
            "do tasks on the table. You can only pick up one object at a time."
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_objects",
                    "description": "Detect objects in the field of view of the camera",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_classes": {
                                "type": "string",
                                "description": (
                                    "Object classes to detect, space ",
                                    "and dot separated. For example: ",
                                    "horses . rivers . plain",
                                ),
                            }
                        },
                        "required": ["object_classes"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_action",
                    "description": "Run an action on an object.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "pick_object",
                                    "move_above_object_and_release",
                                ],
                            },
                            "object_index": {
                                "type": "int",
                                "description": "index of the object in the list of detected objects to execute the action for.",
                            },
                        },
                        "required": ["action", "object"],
                    },
                },
            },
        ],
        model="gpt-4-turbo",
    )
