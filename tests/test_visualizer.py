
import numpy as np

from src.visualizer import draw_results


def test_draw_matched_face():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    results = [{
        "box": [50, 50, 80, 80],
        "label": "Matched",
        "similarity": 0.95,
    }]

    output = draw_results(frame.copy(), results)

    assert output.shape == frame.shape
    assert np.array_equal(output[50, 50], [0, 255, 0])


def test_draw_unknown_face():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    results = [{
        "box": [50, 50, 80, 80],
        "label": "Unknown",
        "similarity": 0.35,
    }]

    output = draw_results(frame.copy(), results)

    assert np.array_equal(output[50, 50], [0, 0, 255])


def test_no_faces():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    output = draw_results(frame.copy(), [])

    assert np.array_equal(output, frame)


def test_box_outside_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    results = [{
        "box": [-20, -20, 60, 60],
        "label": "Matched",
        "similarity": 0.85,
    }]

    output = draw_results(frame.copy(), results)

    assert output.shape == frame.shape
    assert np.array_equal(output[0, 0], [0, 255, 0])
