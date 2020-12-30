import cv2
import numpy as np

from python_eulerian_video_magnification.metadata import MetaData
from python_eulerian_video_magnification.mode import Mode


class Magnify:
    def __init__(self, data: MetaData):
        self._data = data

    def load_video(self, grayscale=False) -> (np.ndarray, int):
        cap = cv2.VideoCapture(self._in_file_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if not grayscale:
            video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        else:
            video_tensor = np.zeros((frame_count, height, width), dtype='float')

        x = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                if grayscale:
                    video_tensor[x] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    video_tensor[x] = frame
                x += 1
            else:
                break
        return video_tensor, fps

    def save_video(self, video_tensor: np.ndarray) -> None:
        # four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        [height, width] = video_tensor[0].shape[0:2]
        writer = cv2.VideoWriter(self._out_file_name, four_cc, 30, (width, height), isColor=True)
        for i in range(0, video_tensor.shape[0]):
            writer.write(cv2.convertScaleAbs(video_tensor[i]))
        writer.release()

    def do_magnify(self) -> None:
        grayscale = self._mode == Mode.RIESZ.name
        grayscale = False
        tensor, fps = self.load_video(grayscale)
        video_tensor = self._magnify_impl(tensor, fps)
        self.save_video(video_tensor)
        self._data.save_meta_data()

    def _magnify_impl(self, tensor: np.ndarray, fps: int) -> np.ndarray:
        raise NotImplementedError("This should be overwritten!")

    @property
    def _low(self) -> float:
        return self._data['low']

    @property
    def _high(self) -> float:
        return self._data['high']

    @property
    def _levels(self) -> int:
        return self._data['levels']

    @property
    def _amplification(self) -> float:
        return self._data['amplification']

    @property
    def _in_file_name(self) -> str:
        return self._data['file']

    @property
    def _out_file_name(self) -> str:
        return self._data['target']
    @property
    def _mode(self) -> str:
        return self._data['mode']
