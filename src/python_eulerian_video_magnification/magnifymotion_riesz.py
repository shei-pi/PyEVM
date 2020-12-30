import cv2
import numpy as np

from python_eulerian_video_magnification.filter import butter_bandpass_filter
from python_eulerian_video_magnification.magnify import Magnify
from python_eulerian_video_magnification.pyramid import laplacian_video
from python_eulerian_video_magnification.riesz_pyramid import riesz_video


class MagnifyMotionRiesz(Magnify):
    def _magnify_impl(self, tensor: np.ndarray, fps: int) -> np.ndarray:
        riesz_video_list = riesz_video(tensor, amplification_factor= self._amplification, 
                                            low_cutoff=self._low, high_cutoff=self._high, 
                                            sampling_rate=fps, levels=self._levels)
        recon = self._reconstruct_from_tensor_list(riesz_video_list)
        return tensor + recon
        # return recon
        
    def _reconstruct_from_tensor_list(self, filter_tensor_list):
        final = np.zeros(filter_tensor_list[-1].shape)
        for i in range(filter_tensor_list[0].shape[0]):
            up = filter_tensor_list[0][i]
            for n in range(self._levels - 1):
                up = cv2.pyrUp(up) + filter_tensor_list[n + 1][i]
            final[i] = up
        return final
