# /workspace/tools/web/manager.py
import time
import threading
import urllib.request

class WebServerManager:
    """uvicorn FastAPI 서버를 백그라운드 스레드로 올리고 관리"""
    def __init__(self):
        self.web_thread = None
        self.web_port = 8000
        self.web_url = f"http://localhost:{self.web_port}"

    def _port_in_use(self, port: int) -> bool:
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _probe_healthz(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=0.6) as r:
                return r.read(2) == b"ok"
        except Exception:
            return False

    def _find_free_port(self, start=8000, end=8100) -> int:
        for p in range(start, end + 1):
            if not self._port_in_use(p):
                return p
        raise RuntimeError("No free port found in range 8000-8100")

    def start_web_server(self) -> bool:
        """웹 서버 시작"""
        try:
            # 기존 서버 재사용
            if self._port_in_use(self.web_port):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"♻️ 기존 웹 서버 재사용: {self.web_url}")
                    return True
                else:
                    new_port = self._find_free_port()
                    print(f"⚠️ 포트 {self.web_port} 사용중(헬스 실패). {new_port}로 변경")
                    self.web_port = new_port
                    self.web_url = f"http://localhost:{self.web_port}"

            print("🌐 웹 서버 시작 중...")

            def _run():
                import uvicorn
                from .server import app
                uvicorn.run(app, host="0.0.0.0", port=self.web_port, log_level="info")

            self.web_thread = threading.Thread(target=_run, daemon=True)
            self.web_thread.start()

            # 헬스체크
            for _ in range(40):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"✅ 웹 서버 시작 완료: {self.web_url}")
                    return True
                time.sleep(0.2)

            print("❌ 웹 서버 시작 확인 실패(헬스 타임아웃)")
            return False

        except Exception as e:
            print(f"❌ 웹 서버 시작 오류: {e}")
            return False

    def open_browser(self):
        try:
            import webbrowser
            print("🌐 웹 브라우저 열기...")
            webbrowser.open(self.web_url)
            print(f"✅ 브라우저에서 {self.web_url} 열림")
        except Exception as e:
            print(f"❌ 브라우저 열기 실패: {e}")
            print(f"수동 접속: {self.web_url}")

    def stop_web_server(self):
        print("ℹ️ uvicorn 스레드는 프로세스 종료 시 함께 정리됩니다.")
