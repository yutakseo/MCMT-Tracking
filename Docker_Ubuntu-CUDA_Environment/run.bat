@echo off
setlocal enableextensions enabledelayedexpansion

rem ===== 설정 =====
set "DOCKER_IMAGE=aei_with_tracking"
set "CONTAINER_NAME=MCMT-container"

rem 스크립트/프로젝트 경로
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "PROJECT_ROOT=%%~fI"
set "DOCKERFILE_PATH=%SCRIPT_DIR%\Dockerfile"
set "DATASETS_FILE=%SCRIPT_DIR%\___DATASETS___.list"

rem 기본값
set "VOLUME_DIR=%CD%"
set "DEBUG=0"

rem ===== 인자 파싱 (-v 경로, -debug) =====
:parse_args
if "%~1"=="" goto after_parse
if /I "%~1"=="-v" (
  if "%~2"=="" ( echo [ERROR] -v requires a path & exit /b 1 )
  set "VOLUME_DIR=%~2"
  shift & shift & goto parse_args
) else if /I "%~1"=="--volume" (
  if "%~2"=="" ( echo [ERROR] --volume requires a path & exit /b 1 )
  set "VOLUME_DIR=%~2"
  shift & shift & goto parse_args
) else if /I "%~1"=="-debug" (
  set "DEBUG=1"
  shift & goto parse_args
) else (
  echo [ERROR] Unknown parameter: %~1
  exit /b 1
)
:after_parse

for %%I in ("%VOLUME_DIR%") do set "VOLUME_DIR=%%~fI"
set "VOLUME_FLAGS=-v "%VOLUME_DIR%:/workspace""

if "%DEBUG%"=="1" echo [DEBUG] VOLUME_DIR=%VOLUME_DIR%

rem ===== 1) 이미지 빌드 (없으면) =====
docker image inspect "%DOCKER_IMAGE%" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Building Docker image: %DOCKER_IMAGE%
  if not exist "%DOCKERFILE_PATH%" (
    echo [ERROR] Dockerfile not found: %DOCKERFILE_PATH%
    exit /b 1
  )
  docker build -t "%DOCKER_IMAGE%" -f "%DOCKERFILE_PATH%" "%PROJECT_ROOT%" || goto :err
  echo [INFO] Docker image built: %DOCKER_IMAGE%
) else (
  echo [INFO] Docker image already exists: %DOCKER_IMAGE%
)

rem ===== 2) 기존 컨테이너 제거 =====
set "CID="
for /f "usebackq delims=" %%C in (`docker ps -aq -f "name=^%CONTAINER_NAME%^$"`) do set "CID=%%C"
if defined CID (
  docker rm -f "%CONTAINER_NAME%" >nul 2>&1
  echo [INFO] Removed existing container: %CONTAINER_NAME%
)

rem ===== 3) ___DATASETS___.list 마운트 구성 =====
if exist "%DATASETS_FILE%" (
  for /f "usebackq delims=" %%L in ("%DATASETS_FILE%") do (
    set "line=%%L"
    if not "!line!"=="" if not "!line:~0,1!"=="#" (
      for %%P in ("!line!") do set "DATASET_PATH=%%~fP"
      if exist "!DATASET_PATH!" (
        for %%N in ("!DATASET_PATH!") do set "DATASET_NAME=%%~nxN"
        if not "!DATASET_NAME!"=="" (
          set "VOLUME_FLAGS=!VOLUME_FLAGS! -v "!DATASET_PATH!:/workspace/datasets/!DATASET_NAME!/""
        ) else (
          echo [WARN] Skipping invalid dataset name: !DATASET_PATH!
        )
      ) else (
        echo [WARN] Invalid path in %DATASETS_FILE%: !line!
      )
    )
  )
  echo [INFO] Volume mount list parsed from %DATASETS_FILE%
) else (
  echo [INFO] No dataset list file found. Skipping additional mounts.
)

if "%DEBUG%"=="1" echo [DEBUG] Volume flags: %VOLUME_FLAGS%

rem ===== 4) 컨테이너 실행 =====
echo [INFO] Starting container: %CONTAINER_NAME%
docker run -dit --gpus all %VOLUME_FLAGS% --shm-size=64g --restart unless-stopped --name "%CONTAINER_NAME%" "%DOCKER_IMAGE%" /bin/bash || goto :err
echo [INFO] Container started in background: %CONTAINER_NAME%

rem ===== 5) 컨테이너 내부 초기화 =====
set "FOUND="
for /f "usebackq delims=" %%N in (`docker ps --format "{{.Names}}"`) do (
  if "%%N"=="%CONTAINER_NAME%" set "FOUND=1"
)
if not defined FOUND (
  echo [ERROR] Container did not start properly. Aborting init.
  exit /b 1
)

docker exec "%CONTAINER_NAME%" bash -lc "mkdir -p /workspace/datasets" || goto :err
echo [INFO] Created /workspace/datasets in container

docker exec "%CONTAINER_NAME%" bash -lc "ln -sf /opt/requirements.txt /workspace/requirements.txt" || echo [WARN] Could not link /opt/requirements.txt
echo [INFO] Created symbolic link for requirements.txt

echo [INFO] Container is up and ready: %CONTAINER_NAME% ^
🎉
exit /b 0

:err
echo [ERROR] Command failed (errorlevel=%errorlevel%)
exit /b %errorlevel%
