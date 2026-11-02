import csv
import cv2
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.camera import Camera
from src.config import CAMERA_ID, REFERENCE_PATH, SIMILARITY_THRESHOLD
from src.face_recognizer import FaceRecognizer
from src.face_tracker import OpticalFlowTracker

BENCHMARK_SECONDS = 60
RECOGNITION_INTERVAL = 2.0
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'benchmark'


def recognize_frame(recognizer, frame):
    start = time.perf_counter()
    results = recognizer.recognize(frame)
    return results, time.perf_counter() - start


def summarize(name, values, unit='ms'):
    if not values:
        print(f'{name}: No samples')
        return
    print(
        f'{name}: mean={statistics.mean(values):.2f} {unit} | '
        f'median={statistics.median(values):.2f} {unit} | '
        f'min={min(values):.2f} {unit} | '
        f'max={max(values):.2f} {unit} | n={len(values)}'
    )


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latency_stats(values):
    if not values:
        return {'count': 0, 'mean_ms': '', 'median_ms': '', 'min_ms': '', 'max_ms': ''}
    return {
        'count': len(values),
        'mean_ms': statistics.mean(values),
        'median_ms': statistics.median(values),
        'min_ms': min(values),
        'max_ms': max(values),
    }


def main():
    recognizer = FaceRecognizer(threshold=SIMILARITY_THRESHOLD)
    reference = cv2.imread(REFERENCE_PATH)
    if reference is None:
        raise FileNotFoundError(REFERENCE_PATH)
    recognizer.register(reference)

    tracker = OpticalFlowTracker()
    camera = Camera(camera_id=CAMERA_ID)
    if not camera.open():
        raise RuntimeError('Cannot open camera')

    frame_count = 0
    tracking_times = []
    recognition_times = []
    last_recognition_start = 0.0
    recognition_future = None
    start_time = time.perf_counter()
    elapsed = 0.0

    print('Benchmark started')
    print(f'Duration: {BENCHMARK_SECONDS} seconds')
    print('Keep one face visible and move naturally')
    print('Press Q to stop early')

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                loop_start = time.perf_counter()
                if loop_start - start_time >= BENCHMARK_SECONDS:
                    break

                success, frame = camera.read_frame()
                if not success or frame is None:
                    break
                frame_count += 1

                if tracker.is_tracking:
                    t0 = time.perf_counter()
                    tracker.update(frame)
                    tracking_times.append((time.perf_counter() - t0) * 1000)

                if recognition_future is not None and recognition_future.done():
                    results, duration = recognition_future.result()
                    recognition_times.append(duration * 1000)
                    recognition_future = None

                    if results and not tracker.is_tracking:
                        result = max(results, key=lambda item: item['similarity'])
                        tracker.initialize(frame, result['box'])

                now = time.perf_counter()
                if (recognition_future is None
                        and now - last_recognition_start >= RECOGNITION_INTERVAL):
                    recognition_future = executor.submit(
                        recognize_frame, recognizer, frame.copy()
                    )
                    last_recognition_start = now

                display = frame.copy()
                if tracker.is_tracking and tracker.box:
                    x, y, w, h = tracker.box
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Face Recognition Benchmark', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Stop the FPS clock before waiting for any outstanding inference.
            elapsed = time.perf_counter() - start_time
            if recognition_future is not None:
                _, duration = recognition_future.result()
                recognition_times.append(duration * 1000)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    fps = frame_count / elapsed if elapsed > 0 else 0.0
    print('\n========== BENCHMARK RESULTS ==========')
    print(f'Duration: {elapsed:.2f} s')
    print(f'Frames: {frame_count}')
    print(f'Average display loop FPS: {fps:.2f}')
    summarize('Tracking latency', tracking_times)
    summarize('Recognition latency', recognition_times)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUTPUT_DIR / 'tracking_latency.csv',
        ['sample', 'latency_ms'],
        [{'sample': i, 'latency_ms': value} for i, value in enumerate(tracking_times, 1)]
    )
    write_csv(
        OUTPUT_DIR / 'recognition_latency.csv',
        ['sample', 'latency_ms'],
        [{'sample': i, 'latency_ms': value} for i, value in enumerate(recognition_times, 1)]
    )
    summary_rows = [
        {'metric': 'duration_s', 'value': elapsed},
        {'metric': 'frames', 'value': frame_count},
        {'metric': 'avg_fps', 'value': fps},
    ]
    for prefix, values in [('tracking', tracking_times), ('recognition', recognition_times)]:
        summary_rows.extend(
            {'metric': f'{prefix}_{key}', 'value': value}
            for key, value in latency_stats(values).items()
        )
    write_csv(OUTPUT_DIR / 'summary.csv', ['metric', 'value'], summary_rows)
    print(f'CSV saved to: {OUTPUT_DIR}')
    print('=======================================')


if __name__ == '__main__':
    main()