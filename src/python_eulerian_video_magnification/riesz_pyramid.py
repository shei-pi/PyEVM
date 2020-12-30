import cv2
import numpy as np

from scipy import ndimage as ndi
from scipy import signal

from python_eulerian_video_magnification.pyramid import build_laplacian_pyramid

from riesz_pyr.riesz_pyramid import laplacian_to_riesz, riesz_to_spherical, riesz_spherical_to_laplacian 
from riesz_pyr.rp_laplacian_like import build_laplacian, collapse_laplacian


def riesz_video( video_tensor, amplification_factor, low_cutoff, high_cutoff,
                                        sampling_rate, levels):

    nyquist_frequency = sampling_rate/2.0

    temporal_filter_order = 1

    [butter_b, butter_a] = signal.butter(temporal_filter_order, [low_cutoff/nyquist_frequency,
                            high_cutoff/nyquist_frequency],btype="bandpass", fs=sampling_rate)

    gaussian_kernel_sd = 2
    gaussian_kernel = get_gaussian_kernel(gaussian_kernel_sd)
    curr_frame_n = 0
    #INITIALIZATION
    previous_frame = video_tensor[curr_frame_n]
    
    previous_laplacian_pyramid, \
    previous_riesz_x, \
    previous_riesz_y = compute_riesz_pyramid(previous_frame,levels)

    number_of_levels = levels-1

    phase_cos, phase_sin, register0_cos, \
        register1_cos, register0_sin, register1_sin = \
            [[np.zeros_like(previous_laplacian_pyramid[k]) for k in range(number_of_levels)]]*6
    curr_frame_n+=1
    motion_magnified_laplacian_pyramids_list=[]
    for k in range(number_of_levels+1):
        motion_magnified_laplacian_pyramids_list.append(np.zeros((video_tensor.shape[0],
                                                        previous_laplacian_pyramid[k].shape[0],
                                                        previous_laplacian_pyramid[k].shape[1],
                                                        #3
        ))) 
    for idx,current_frame in enumerate(video_tensor[curr_frame_n:]):
        
        current_laplacian_pyramid, current_riesz_x, current_riesz_y = compute_riesz_pyramid(current_frame, levels)

        for k in range(number_of_levels):#goes from k=0,1..(number_of_levels-1)
            [phase_difference_cos, phase_difference_sin, amplitude] = \
                            compute_phase_difference_and_amplitude(current_laplacian_pyramid[k],
                                                                    current_riesz_x[k],
                                                                            current_riesz_y[k],
                                                                            previous_laplacian_pyramid[k],
                                                                            previous_riesz_x[k],
                                                                            previous_riesz_y[k])
            phase_cos[k] += phase_difference_cos
            phase_sin[k] += phase_difference_sin

            [phase_filtered_cos, register0_cos[k], register1_cos[k]] = iir_temporal_filter(butter_b,butter_a, phase_cos[k], register0_cos[k], 
                                                                                            register1_cos[k])
            [phase_filtered_sin, register0_sin[k], register1_sin[k]] = iir_temporal_filter(butter_b,butter_a, phase_sin[k], register0_sin[k], 
                                                                                            register1_sin[k])
            
            phase_filtered_cos = amplitude_weighted_blur(phase_filtered_cos, amplitude, gaussian_kernel)
            phase_filtered_sin = amplitude_weighted_blur(phase_filtered_sin, amplitude, gaussian_kernel)


            phase_magnified_filtered_cos = amplification_factor + phase_filtered_cos

            phase_magnified_filtered_sin = amplification_factor + phase_filtered_sin

            mm_lap_pyr  =   phase_shift_coefficient_real_part(current_laplacian_pyramid[k],
                                                                                        current_riesz_x[k],
                                                                                        current_riesz_y[k],
                                                                                        phase_magnified_filtered_cos,
                                                                                        phase_magnified_filtered_sin)

            motion_magnified_laplacian_pyramids_list[k][idx]=mm_lap_pyr                                                                                        
           
        
        motion_magnified_laplacian_pyramids_list[number_of_levels][idx]=current_laplacian_pyramid[number_of_levels]

        previous_laplacian_pyramid = current_laplacian_pyramid
        previous_riesz_x = current_riesz_x
        previous_riesz_y = current_riesz_y
    
    return motion_magnified_laplacian_pyramids_list

def compute_riesz_pyramid(grayscale_frame, levels):
    number_of_levels = levels
    # laplacian_pyramid = build_laplacian_pyramid(grayscale_frame, number_of_levels)
    laplacian_pyramid = build_laplacian(grayscale_frame)

    # kernel_x = np.array([[0.0,0.0,0.0],
    #                      [0.5,0.0,-0.5],
    #                      [0.0,0.0,0.0]])   
    # kernel_y = np.array([[0.0,0.5,0.0],
    #                      [0.0,0.0,0.0],
    #                      [0.0,-0.5,0.0]])   
    riesz_band_filter = np.asarray([[-0.12,0,0.12],[-0.34, 0, 0.34],[-0.12,0,0.12]])
    kernel_x = riesz_band_filter
    kernel_y = riesz_band_filter.T
    riesz_x=[]
    riesz_y=[]
    for lp in laplacian_pyramid:
        riesz_x.append(cv2.filter2D(lp,-1,kernel_x))
        riesz_y.append(cv2.filter2D(lp,-1,kernel_y))
    return laplacian_pyramid, riesz_x, riesz_y

def compute_phase_difference_and_amplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y):

    q_conj_prod_real = np.multiply(current_real, previous_real) + np.multiply(current_x, previous_x) + np.multiply(current_y, previous_y)

    q_conj_prod_x = -np.multiply(current_real, previous_x) + np.multiply(previous_real, current_x)
    q_conj_prod_y = -np.multiply(current_real, previous_y) + np.multiply(previous_real, current_y)

    q_conj_prod_amplitude = np.sqrt(np.power(q_conj_prod_real,2)+np.power(q_conj_prod_x,2)+np.power(q_conj_prod_y,2))
    phase_difference = np.arccos(np.divide(q_conj_prod_real, q_conj_prod_amplitude))

    conj_prod_rms =  np.sqrt(np.power(q_conj_prod_x,2)+np.power(q_conj_prod_y,2))
    cos_orientation = np.divide(q_conj_prod_x,conj_prod_rms)
    sin_orientation = np.divide(q_conj_prod_y,conj_prod_rms)

    phase_difference_cos = np.multiply(phase_difference, cos_orientation)
    phase_difference_sin = np.multiply(phase_difference, sin_orientation)

    amplitude = np.sqrt(q_conj_prod_amplitude)

    return phase_difference_cos, phase_difference_sin, amplitude

def iir_temporal_filter(b, a, phase, register0, register1):
    temporally_filtered_phase = b[0] * phase * register0
    r0 = b[1] * phase + register1 - a[1] * temporally_filtered_phase
    r1 = b[2] * phase             - a[2] * temporally_filtered_phase

    return temporally_filtered_phase, r0, r1

def amplitude_weighted_blur(temporally_filtered_phase, amplitude, blur_kernel):

    denominator = ndi.convolve(amplitude, blur_kernel)
    numerator   = ndi.convolve(np.multiply(temporally_filtered_phase,amplitude),blur_kernel)

    spatially_smooth_temporally_filtered_phase = np.divide(numerator, denominator)

    return spatially_smooth_temporally_filtered_phase

def phase_shift_coefficient_real_part(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin):

    phase_magnitude = np.sqrt(np.power(phase_cos,2)+np.power(phase_sin,2))

    exp_phase_real = np.cos(phase_magnitude)

    denominator = np.multiply(phase_magnitude,np.sin(phase_magnitude))
    exp_phase_x     = np.divide(phase_cos, denominator)
    exp_phase_y     = np.divide(phase_sin, denominator)
    result = np.multiply(exp_phase_real, riesz_real)- np.multiply(exp_phase_x, riesz_x) - np.multiply(exp_phase_y, riesz_y)

    return result
#REVISAR!!!! https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy#43346070
def get_gaussian_kernel(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)
    