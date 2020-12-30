import numpy

def symmetrical_boundary(img):
	#manually set up a symmetrical boundary condition so we can use fftconvolve
	#but avoid edge effects
	(h,w) = img.shape
	imgsymm = numpy.empty((h*2, w*2))
	imgsymm[h/2:-(h+1)/2, w/2:-(w+1)/2] = img.copy()
	imgsymm[0:h/2, 0:w/2] = img[0:h/2, 0:w/2][::-1,::-1].copy()
	imgsymm[-(h+1)/2:, -(w+1)/2:] = img[-(h+1)/2:, -(w+1)/2:][::-1,::-1].copy()
	imgsymm[0:h/2, -(w+1)/2:] = img[0:h/2, -(w+1)/2:][::-1,::-1].copy()
	imgsymm[-(h+1)/2:, 0:w/2] = img[-(h+1)/2:, 0:w/2][::-1,::-1].copy()
	imgsymm[h/2:-(h+1)/2, 0:w/2] = img[:, 0:w/2][:,::-1].copy()
	imgsymm[h/2:-(h+1)/2, -(w+1)/2:] = img[:, -(w+1)/2:][:,::-1].copy()
	imgsymm[0:h/2, w/2:-(w+1)/2] = img[0:h/2, :][::-1,:].copy()
	imgsymm[-(h+1)/2:, w/2:-(w+1)/2] = img[-(h+1)/2:, :][::-1,:].copy()
	return imgsymm