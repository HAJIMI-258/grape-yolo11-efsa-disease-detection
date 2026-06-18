from __future__ import annotations

import base64
import os
from pathlib import Path

import paramiko

HOST = os.environ.get("GRAPE_REMOTE_HOST")
USER = os.environ.get("GRAPE_REMOTE_USER", "Administrator")
PASSWORD = os.environ.get("GRAPE_REMOTE_PASSWORD")
LOCAL = Path(__file__).resolve().parent
REMOTE = r"D:\grape_combo"
TASK_NAME = "TrainHighSource7YOLO11STeacher"


def run_ps(client: paramiko.SSHClient, script: str, timeout: int = 900) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    _stdin, stdout, stderr = client.exec_command(f"powershell -NoProfile -EncodedCommand {encoded}", timeout=timeout)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def main() -> None:
    if not HOST or not PASSWORD:
        raise RuntimeError("Set GRAPE_REMOTE_HOST and GRAPE_REMOTE_PASSWORD before connecting to the remote trainer.")

    script_name = "train_yolo11s_teacher_highsource7_region_remote.py"
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
        sftp.put(str(LOCAL / script_name), rf"{REMOTE}\{script_name}")
        batch = rf"""@echo off
setlocal
cd /d {REMOTE}
set PYTHONPATH={REMOTE}\grape-yolo11-efsa-disease-detection;{REMOTE}
echo [%date% %time%] YOLO11s teacher training started > {REMOTE}\train_yolo11s_teacher_task.log
D:\Python311\python.exe {REMOTE}\{script_name} > {REMOTE}\train_yolo11s_teacher_stdout.log 2> {REMOTE}\train_yolo11s_teacher_stderr.log
echo [%date% %time%] YOLO11s teacher training finished code %ERRORLEVEL% >> {REMOTE}\train_yolo11s_teacher_task.log
"""
        with sftp.open(rf"{REMOTE}\run_yolo11s_teacher.bat", "w") as handle:
            handle.write(batch)
        sftp.close()

        ps = rf"""
$ErrorActionPreference='Stop'
if(-not (Test-Path '{REMOTE}\highsource7_region_yolo_v1\data.yaml')) {{ throw 'data config missing' }}
if(-not (Test-Path 'D:\grape_mypfe6\yolo11s.pt')) {{ throw 'D:\grape_mypfe6\yolo11s.pt missing' }}
$runDir='{REMOTE}\runs\yolo11s_highsource7_teacher_img640_e150'
if(Test-Path $runDir) {{ Remove-Item -LiteralPath $runDir -Recurse -Force }}
cmd /c "schtasks /End /TN {TASK_NAME} 2>nul"
cmd /c "schtasks /Delete /TN {TASK_NAME} /F 2>nul"
$start=(Get-Date).AddMinutes(1)
$st=$start.ToString('HH:mm')
$sd=$start.ToString('yyyy/MM/dd')
cmd /c "schtasks /Create /TN {TASK_NAME} /SC ONCE /SD $sd /ST $st /TR {REMOTE}\run_yolo11s_teacher.bat /RL HIGHEST /F"
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
