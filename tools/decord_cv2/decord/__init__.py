"""decord's VideoReader on OpenCV, for platforms without a decord wheel (aarch64 Linux).

Covers what VBench 0.1.5 imports and calls: VideoReader(uri, ctx, width, height, num_threads), len(), get_avg_fps(), get_batch(),
indexing, cpu() and decord.bridge.set_bridge(). Frames are RGB uint8, shape (T, H, W, 3), like decord's. Other decord features
(audio, GPU contexts, seeking by timestamp) are not provided.
"""
import cv2
import numpy as np

from . import bridge

try:
    import torch
except ImportError:
    torch = None

__version__ = "0.6.0+cv2"


def cpu(device_id: int = 0) -> tuple[str, int]:
    return ("cpu", device_id)


class NDArray:
    """The result of get_batch with the native bridge: decord's own array type exposes asnumpy()."""

    def __init__(self, array: np.ndarray):
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, index):
        return NDArray(self._array[index]) if isinstance(index, slice) else self._array[index]


class VideoReader:
    def __init__(self, uri, ctx=None, width: int = -1, height: int = -1, num_threads: int = 0, fault_tol: int = -1):
        self._capture = cv2.VideoCapture(str(uri))
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open video {uri}")
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._size = (width, height) if width > 0 and height > 0 else None

    def __len__(self) -> int:
        return self._length

    def get_avg_fps(self) -> float:
        return self._fps

    def _frames(self, indices: list[int]) -> np.ndarray:
        wanted = [index + self._length if index < 0 else index for index in indices]
        assert all(0 <= index < self._length for index in wanted), f"frame index outside 0..{self._length - 1}"
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        found: dict[int, np.ndarray] = {}
        last = max(wanted)
        for position in range(last + 1):
            ok, frame = self._capture.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {position}, before frame {last}")
            if position in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                found[position] = cv2.resize(frame, self._size) if self._size else frame
        return np.stack([found[index] for index in wanted])

    def get_batch(self, indices):
        array = self._frames([int(index) for index in indices])
        mode = bridge.get_bridge()
        if mode == "torch":
            assert torch is not None, "the torch bridge needs torch installed"
            return torch.from_numpy(array)
        if mode == "numpy":
            return array
        return NDArray(array)

    def __getitem__(self, index: int):
        return self.get_batch([index])[0]
