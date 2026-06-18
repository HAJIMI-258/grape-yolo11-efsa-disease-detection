from __future__ import annotations

import base64
import contextlib
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("GRAPE_REMOTE_HOST")
USER = os.environ.get("GRAPE_REMOTE_USER", "Administrator")
PASSWORD = os.environ.get("GRAPE_REMOTE_PASSWORD")
LOCAL = Path(r"C:\Users\a1366\Desktop\grape data\grape-yolo11-efsa-disease-detection")
REMOTE = r"D:\grape_combo"
TASK_NAME = "TrainHighSource7Width20GEWGapKD"


def run_ps(client: paramiko.SSHClient, script: str, timeout: int = 900) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    _stdin, stdout, stderr = client.exec_command(f"powershell -NoProfile -EncodedCommand {encoded}", timeout=timeout)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def main() -> None:
    if not HOST or not PASSWORD:
        raise RuntimeError("Set GRAPE_REMOTE_HOST and GRAPE_REMOTE_PASSWORD before connecting to the remote trainer.")

    uploads = [
        ("ultralytics/nn/modules/block.py", "grape-yolo11-efsa-disease-detection\\ultralytics\\nn\\modules\\block.py"),
        ("ultralytics/nn/modules/efsa.py", "grape-yolo11-efsa-disease-detection\\ultralytics\\nn\\modules\\efsa.py"),
        (
            "ultralytics/nn/modules/__init__.py",
            "grape-yolo11-efsa-disease-detection\\ultralytics\\nn\\modules\\__init__.py",
        ),
        ("ultralytics/nn/tasks.py", "grape-yolo11-efsa-disease-detection\\ultralytics\\nn\\tasks.py"),
        ("ultralytics/utils/loss.py", "grape-yolo11-efsa-disease-detection\\ultralytics\\utils\\loss.py"),
        (
            "scripts/highsource7/train_yolo11n_width20_gapkd_v12_highsource7_region_remote.py",
            "train_yolo11n_width20_gapkd_v12_highsource7_region_remote.py",
        ),
        (
            "scripts/highsource7/train_yolo11n_width20_gew_gapkd_highsource7_region_remote.py",
            "train_yolo11n_width20_gew_gapkd_highsource7_region_remote.py",
        ),
        ("configs/models/yolo11n_width20_gew_highsource7_region.yaml", "yolo11n_width20_gew_highsource7_region.yaml"),
        ("docs/HIGHSOURCE7_FAIR_PROTOCOL.md", "highsource7_fair_protocol.md"),
    ]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=20,
        auth_timeout=20,
        banner_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    try:
        sftp = client.open_sftp()
        for local_name, remote_name in uploads:
            print(f"upload {local_name}")
            remote_dir = str(Path(rf"{REMOTE}\{remote_name}").parent)
            with contextlib.suppress(OSError):
                sftp.mkdir(remote_dir)
            sftp.put(str(LOCAL / local_name), rf"{REMOTE}\{remote_name}")

        batch = rf"""@echo off
setlocal
cd /d {REMOTE}
echo [%date% %time%] width20 GEW-gapKD training started > {REMOTE}\train_width20_gew_gapkd_region_task.log
D:\Python311\python.exe {REMOTE}\train_yolo11n_width20_gew_gapkd_highsource7_region_remote.py > {REMOTE}\train_width20_gew_gapkd_region_stdout.log 2> {REMOTE}\train_width20_gew_gapkd_region_stderr.log
echo [%date% %time%] width20 GEW-gapKD training finished code %ERRORLEVEL% >> {REMOTE}\train_width20_gew_gapkd_region_task.log
"""
        with sftp.open(rf"{REMOTE}\run_width20_gew_gapkd_region.bat", "w") as handle:
            handle.write(batch)
        sftp.close()

        smoke_py = rf"""
import sys
from pathlib import Path
root = Path(r"{REMOTE}\grape-yolo11-efsa-disease-detection")
sys.path.insert(0, str(root))
sys.path.insert(0, r"{REMOTE}")
from ultralytics import YOLO
import train_yolo11n_width20_gew_gapkd_highsource7_region_remote as gew
student = YOLO(r"{REMOTE}\yolo11n_width20_gew_highsource7_region.yaml")
print("student params", sum(p.numel() for p in student.model.parameters()))
print("trainer", gew.GEWGapDistillTrainer.__name__)
"""
        ps = rf"""
$ErrorActionPreference='Stop'
Write-Host '--- smoke width20 GEW-gapKD ---'
$env:PYTHONPATH='{REMOTE}\grape-yolo11-efsa-disease-detection;{REMOTE}'
@'
{smoke_py}
'@ | D:\Python311\python.exe -
if($LASTEXITCODE -ne 0) {{ throw 'smoke build failed' }}

Write-Host '--- clear previous GEW-gapKD run ---'
$runDir='{REMOTE}\runs\yolo11n_width20_gew_gapkd_highsource7_region_img640_e150'
if(Test-Path $runDir) {{ Remove-Item -LiteralPath $runDir -Recurse -Force }}

Write-Host '--- schedule width20 GEW-gapKD training ---'
cmd /c "schtasks /End /TN {TASK_NAME} 2>nul"
cmd /c "schtasks /Delete /TN {TASK_NAME} /F 2>nul"
$start=(Get-Date).AddMinutes(1)
$st=$start.ToString('HH:mm')
$sd=$start.ToString('yyyy/MM/dd')
cmd /c "schtasks /Create /TN {TASK_NAME} /SC ONCE /SD $sd /ST $st /TR {REMOTE}\run_width20_gew_gapkd_region.bat /RL HIGHEST /F"
Get-ScheduledTask -TaskName {TASK_NAME} | Select-Object TaskName,State | Format-Table
"""
        out, err = run_ps(client, ps, timeout=1500)
        print(out[-20000:].encode("gbk", "replace").decode("gbk"))
        if err.strip():
            print("STDERR")
            print(err[-8000:].encode("gbk", "replace").decode("gbk"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
