"""Day-2 ML basics: SEE object detection work, before you train anything.

This downloads a tiny pretrained YOLO11 model (trained on the generic COCO dataset:
people, cars, buses...) and runs it on a sample image. It detects COCO objects, NOT
drawing parts yet — the point is to watch the detect→boxes flow and learn the API you'll
reuse on your own model.

Run:
    pip install ultralytics
    python scripts/try_pretrained.py
    # optional: python scripts/try_pretrained.py path/to/your_image.jpg
"""
import sys


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install first:  pip install ultralytics")
        sys.exit(1)

    # 'yolo11n.pt' = nano model, auto-downloaded on first run (~5 MB).
    model = YOLO("yolo11n.pt")

    # Default sample image (a bus + people). Or pass your own as arg 1.
    source = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/bus.jpg"

    # predict() runs one forward pass -> Results. save=True writes an annotated image.
    results = model.predict(source, conf=0.25, save=True, verbose=False)

    print("\n--- what the model found ---")
    for r in results:
        names = r.names  # {class_index: class_name}
        for box in r.boxes:
            cls = names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
            print(f"{cls:12s} conf={conf:.2f}  box=({x1},{y1})-({x2},{y2})")
        print(f"\nAnnotated image saved under: {r.save_dir}")

    print(
        "\nThat is exactly the data shape src/detect.py returns. "
        "After you train on your drawings, swap 'yolo11n.pt' for your models/best.pt "
        "and the classes become title_block / dimension / gdt_symbol / ..."
    )


if __name__ == "__main__":
    main()
