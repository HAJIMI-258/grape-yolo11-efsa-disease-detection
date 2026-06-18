from __future__ import annotations

import base64
import os
from pathlib import Path

import paramiko


HOST = os.environ.get("GRAPE_REMOTE_HOST")
USER = os.environ.get("GRAPE_REMOTE_USER", "Administrator")
PASSWORD = os.environ.get("GRAPE_REMOTE_PASSWORD")
LOCAL = Path(r"C:\Users\a1366\Desktop\grape data")
REMOTE = r"D:\grape_combo"
TASK_NAME = "TrainHighSource7YOLO11nGapKDFull"


def run_ps(client: paramiko.SSHClient, script: str, timeout: int = 900) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    _stdin, stdout, stderr = client.exec_command(f"powershell -NoProfile -EncodedCommand {encoded}", timeout=timeout)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def main() -> None:
    if not HOST or not PASSWORD:
        raise RuntimeError("Set GRAPE_REMOTE_HOST and GRAPE_REMOTE_PASSWORD before connecting to the remote trainer.")

    uploads = [
        "train_yolo11n_gapkd_full_highsource7_region_remote.py",
        "highsource7_fair_protocol.md",
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
        for name in uploads:
            print(f"upload {name}")
            sftp.put(str(LOCAL / name), rf"{REMOTE}\{name}")

        batch = rf"""@echo off
setlocal
cd /d {REMOTE}
echo [%date% %time%] full YOLO11n gapKD training started > {REMOTE}\train_yolo11n_gapkd_full_region_task.log
D:\Python311\python.exe {REMOTE}\train_yolo11n_gapkd_full_highsource7_region_remote.py > {REMOTE}\train_yolo11n_gapkd_full_region_stdout.log 2> {REMOTE}\train_yolo11n_gapkd_full_region_stderr.log
echo [%date% %time%] full YOLO11n gapKD training finished code %ERRORLEVEL% >> {REMOTE}\train_yolo11n_gapkd_full_region_task.log
"""
        with sftp.open(rf"{REMOTE}\run_yolo11n_gapkd_full_region.bat", "w") as handle:
            handle.write(batch)
        sftp.close()

        smoke_py = rf"""
import sys
from pathlib import Path
root = Path(r"{REMOTE}\grape-yolo11-efsa-disease-detection")
sys.path.insert(0, str(root))
sys.path.insert(0, r"{REMOTE}")
from ultralytics import YOLO
import train_yolo11n_gapkd_full_highsource7_region_remote as fullkd
student = YOLO(r"D:\grape_mypfe6\yolo11n.pt")
teacher = YOLO(r"{REMOTE}\runs\yolo11n_highsource7_region_fast_img640_e150\weights\best.pt")
print("student raw params", sum(p.numel() for p in student.model.parameters()))
print("teacher raw params", sum(p.numel() for p in teacher.model.parameters()))
print("trainer", fullkd.FullGapDistillTrainer.__name__)
"""
        ps = rf"""
$ErrorActionPreference='Stop'
Write-Host '--- smoke full gapKD ---'
$env:PYTHONPATH='{REMOTE}\grape-yolo11-efsa-disease-detection;{REMOTE}'
@'
{smoke_py}
'@ | D:\Python311\python.exe -
if($LASTEXITCODE -ne 0) {{ throw 'smoke build failed' }}

Write-Host '--- clear previous full gapKD run ---'
$runDir='{REMOTE}\runs\yolo11n_gapkd_full_highsource7_region_img640_e150'
if(Test-Path $runDir) {{ Remove-Item -LiteralPath $runDir -Recurse -Force }}

Write-Host '--- schedule full YOLO11n gapKD training ---'
cmd /c "schtasks /End /TN {TASK_NAME} 2>nul"
cmd /c "schtasks /Delete /TN {TASK_NAME} /F 2>nul"
$start=(Get-Date).AddMinutes(1)
$st=$start.ToString('HH:mm')
$sd=$start.ToString('yyyy/MM/dd')
cmd /c "schtasks /Create /TN {TASK_NAME} /SC ONCE /SD $sd /ST $st /TR {REMOTE}\run_yolo11n_gapkd_full_region.bat /RL HIGHEST /F"
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
