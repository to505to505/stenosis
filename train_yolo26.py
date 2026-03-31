import sys
# Remove CWD from sys.path to avoid local 'ultralytics/' folder shadowing the installed package
sys.path = [p for p in sys.path if p not in ("", ".", "/home/dsa/stenosis")]

from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(
    data="/home/dsa/stenosis/data/stenosis_arcade/data.yaml",
    epochs=100,
    batch=8,
    imgsz=512,
    device=0,
    workers=8,
    optimizer="SGD",
    patience=100,
    project="runs/detect",
    name="stenosis_arcade_yolo26m_same_hyps",
    amp=True,
    plots=True,
    verbose=True,
    # Learning rate
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    # Warmup
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.9,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.3,
)
