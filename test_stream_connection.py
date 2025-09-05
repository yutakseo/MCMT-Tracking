#!/usr/bin/env python3
import sys
sys.path.append("/workspace")

from MCMT_engine.cam_stream import CamMJPEG
from app_web import set_webviz, set_cam_streams
from tools.webviz import WebPlanViz
import time

print("🔗 카메라 스트림 연결 테스트 시작...")

# WebPlanViz 초기화
viz = WebPlanViz(plan_path="/workspace/assets/250904_homograph_coordinate-plane2.jpg", show_cam_points=False, fps_limit=12.0)
set_webviz(viz)
print("✅ WebPlanViz 연결 완료")

# CamMJPEG 스트림 초기화
streams = {}
for i in range(3):
    try:
        stream = CamMJPEG(name=f"cam{i+1}", url="rtsp://210.99.70.120:1935/live/cctv001.stream", width=480).start()
        streams[f"cam{i+1}"] = stream
        print(f"✅ Stream {i+1} 초기화 완료")
    except Exception as e:
        print(f"❌ Stream {i+1} 초기화 실패: {e}")

set_cam_streams(streams)
print(f"✅ {len(streams)}개 카메라 스트림 연결 완료")

print("🎉 모든 스트림이 웹 서버에 연결되었습니다!")
print("🌐 브라우저에서 http://localhost:8000 에 접속하세요")

# 무한 대기
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n종료됨")
