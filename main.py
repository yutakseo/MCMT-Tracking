# /workspace/main.py
import asyncio
from typing import Optional, List
from Thirdparty_tools.web.app import MultiCameraWebApp

async def main(detector_models: Optional[List[str]] = None):
    app = MultiCameraWebApp(detector_models=detector_models, draw_mode="auto")
    await app.run()

if __name__ == "__main__":
    # 원하는 모델 조합 지정
    asyncio.run(main(detector_models=["ultra"]))
