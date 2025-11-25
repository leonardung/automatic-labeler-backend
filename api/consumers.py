import json

from channels.generic.websocket import AsyncWebsocketConsumer


class ImageConsumer(AsyncWebsocketConsumer):
    """
    Legacy websocket endpoint kept for compatibility. The previous tennis-ball
    tracker depended on removed ml_models assets. We now respond immediately
    with a descriptive message so the frontend can handle the missing feature
    gracefully.
    """

    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        _ = json.loads(text_data)
        await self.send(
            text_data=json.dumps(
                {
                    "status": "error",
                    "message": "Automatic tennis-ball tracking has been removed from the backend.",
                }
            )
        )
        await self.close()
