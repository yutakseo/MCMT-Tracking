# /workspace/main.py
import asyncio
from typing import Optional, List
from tools.web.app import MultiCameraWebApp

async def main(detector_models: Optional[List[str]] = None):
    app = MultiCameraWebApp(detector_models=detector_models)
    await app.run()

if __name__ == "__main__":
    # 원하는 모델 조합 지정
    asyncio.run(main(detector_models=["ultra_people", "worker", "vehicle"]))
