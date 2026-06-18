from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO(r"D:\grape_mypfe6\yolo11n.pt")
    model.train(
        data=r"D:\grape_combo\highsource7_region_yolo_v1\data.yaml",
        imgsz=640,
        epochs=150,
        batch=64,
        workers=4,
        device=0,
        pretrained=True,
        patience=0,
        amp=True,
        cache=False,
        seed=42,
        box=8.0,
        cls=0.65,
        dfl=1.7,
        mosaic=0.45,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=20,
        project=r"D:\grape_combo\runs",
        name="yolo11n_highsource7_region_fast_img640_e150",
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
