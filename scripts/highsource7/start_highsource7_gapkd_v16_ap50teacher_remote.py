from __future__ import annotations

import base64
import os
from pathlib import Path

import paramiko


HOST = os.environ.get("GRAPE_REMOTE_HOST")
USER = os.environ.get("GRAPE_REMOTE_USER", "Administrator")
PASSWORD = os.environ.get("GRAPE_REMOTE_PASSWORD")
REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE = r"D:\grape_combo"
TASK_NAME = "TrainHighSource7GapKDV16AP50Teacher"


def run_ps(client: paramiko.SSHClient, script: str, timeout: int = 900) -> tuple[str, str]:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    _stdin, stdout, stderr = client.exec_command(
        f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}",
        timeout=timeout,
    )
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def main() -> None:
    if not HOST or not PASSWORD:
        raise RuntimeError("Set GRAPE_REMOTE_HOST and GRAPE_REMOTE_PASSWORD before connecting to the remote trainer.")

    uploads = [
        ("scripts/highsource7/highsource7_ap50_checkpoint.py", "highsource7_ap50_checkpoint.py"),
        (
            "scripts/highsource7/train_yolo11n_highsource7_region_ap50_teacher_remote.py",
            "train_yolo11n_highsource7_region_ap50_teacher_remote.py",
        ),
        (
            "scripts/highsource7/train_yolo11n_width20_gapkd_v12_highsource7_region_remote.py",
            "train_yolo11n_width20_gapkd_v12_highsource7_region_remote.py",
        ),
        (
            "scripts/highsource7/train_yolo11n_width20_gapkd_v14_highsource7_region_remote.py",
            "train_yolo11n_width20_gapkd_v14_highsource7_region_remote.py",
        ),
        (
            "scripts/highsource7/train_yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_remote.py",
            "train_yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_remote.py",
        ),
        ("scripts/highsource7/yolo11n_width20_highsource7.yaml", "yolo11n_width20_highsource7.yaml"),
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
        for src, dst in uploads:
            print(f"upload {src} -> {dst}")
            sftp.put(str(REPO_ROOT / src), rf"{REMOTE}\{dst}")

        batch = rf"""@echo off
setlocal
cd /d {REMOTE}
echo [%date% %time%] v16 AP50-teacher pipeline started > {REMOTE}\train_gapkd_v16_ap50teacher_task.log
if exist {REMOTE}\runs\yolo11n_highsource7_region_ap50teacher_img640_e150\weights\best_map50.pt (
  echo [%date% %time%] AP50 teacher exists, skipping teacher training >> {REMOTE}\train_gapkd_v16_ap50teacher_task.log
) else (
  echo [%date% %time%] AP50 teacher training started >> {REMOTE}\train_gapkd_v16_ap50teacher_task.log
  D:\Python311\python.exe {REMOTE}\train_yolo11n_highsource7_region_ap50_teacher_remote.py > {REMOTE}\train_yolo11n_ap50_teacher_stdout.log 2> {REMOTE}\train_yolo11n_ap50_teacher_stderr.log
  if errorlevel 1 exit /b %ERRORLEVEL%
)
echo [%date% %time%] v16 student training started >> {REMOTE}\train_gapkd_v16_ap50teacher_task.log
D:\Python311\python.exe {REMOTE}\train_yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_remote.py > {REMOTE}\train_width20_gapkd_v16_ap50teacher_stdout.log 2> {REMOTE}\train_width20_gapkd_v16_ap50teacher_stderr.log
echo [%date% %time%] v16 AP50-teacher pipeline finished code %ERRORLEVEL% >> {REMOTE}\train_gapkd_v16_ap50teacher_task.log
"""
        with sftp.open(rf"{REMOTE}\run_gapkd_v16_ap50teacher.bat", "w") as handle:
            handle.write(batch)
        sftp.close()

        smoke_py = rf"""
import sys
from pathlib import Path
root = Path(r"{REMOTE}\grape-yolo11-efsa-disease-detection")
sys.path.insert(0, str(root))
sys.path.insert(0, r"{REMOTE}")
from ultralytics import YOLO
import train_yolo11n_highsource7_region_ap50_teacher_remote as base
import train_yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_remote as v16
student = YOLO(r"{REMOTE}\yolo11n_width20_highsource7.yaml")
print("baseline trainer", base.AP50BaselineTrainer.__name__)
print("student params", sum(p.numel() for p in student.model.parameters()))
print("v16 trainer", v16.GapFeatureDistillV16Trainer.__name__)
"""
        ps = rf"""
$ErrorActionPreference='Stop'
Write-Host '--- smoke AP50 teacher + v16 ---'
$env:PYTHONPATH='{REMOTE}\grape-yolo11-efsa-disease-detection;{REMOTE}'
@'
{smoke_py}
'@ | D:\Python311\python.exe -
if($LASTEXITCODE -ne 0) {{ throw 'smoke build failed' }}

Write-Host '--- clear previous v16 student run ---'
$studentRun='{REMOTE}\runs\yolo11n_width20_gapkd_v16_ap50teacher_highsource7_region_img640_e150'
if(Test-Path $studentRun) {{ Remove-Item -LiteralPath $studentRun -Recurse -Force }}

Write-Host '--- stop failed old tasks ---'
cmd /c "schtasks /End /TN TrainHighSource7Width20GapKDV15 2>nul"
cmd /c "schtasks /Delete /TN TrainHighSource7Width20GapKDV15 /F 2>nul"

Write-Host '--- schedule AP50 teacher + v16 pipeline ---'
cmd /c "schtasks /End /TN {TASK_NAME} 2>nul"
cmd /c "schtasks /Delete /TN {TASK_NAME} /F 2>nul"
$start=(Get-Date).AddMinutes(1)
$st=$start.ToString('HH:mm')
$sd=$start.ToString('yyyy/MM/dd')
cmd /c "schtasks /Create /TN {TASK_NAME} /SC ONCE /SD $sd /ST $st /TR {REMOTE}\run_gapkd_v16_ap50teacher.bat /RL HIGHEST /F"
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
