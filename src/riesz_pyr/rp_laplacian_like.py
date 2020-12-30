"""
rp_laplacian_like.py
Conversion between image and laplacian-like pyramids
Based on the data structures and methodoligies described in:

Riesz Pyramids for Fast Phase-Based Video Magnification
Neal Wadhwa, Michael Rubinstein, Fredo Durand and William T. Freeman
Computational Photography (ICCP), 2014 IEEE International Conference on 

Copyright (c) 2016 Jack Doerner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
""" 

import numpy, cv2, scipy.signal
from riesz_pyr.rp_boundary import symmetrical_boundary

lowpass = numpy.asarray([
	[-0.0001,   -0.0007,  -0.0023,  -0.0046,  -0.0057,  -0.0046,  -0.0023,  -0.0007,  -0.0001],
	[-0.0007,   -0.0030,  -0.0047,  -0.0025,  -0.0003,  -0.0025,  -0.0047,  -0.0030,  -0.0007],
	[-0.0023,   -0.0047,   0.0054,   0.0272,   0.0387,   0.0272,   0.0054,  -0.0047,  -0.0023],
	[-0.0046,   -0.0025,   0.0272,   0.0706,   0.0910,   0.0706,   0.0272,  -0.0025,  -0.0046],
	[-0.0057,   -0.0003,   0.0387,   0.0910,   0.1138,   0.0910,   0.0387,  -0.0003,  -0.0057],
	[-0.0046,   -0.0025,   0.0272,   0.0706,   0.0910,   0.0706,   0.0272,  -0.0025,  -0.0046],
	[-0.0023,   -0.0047,   0.0054,   0.0272,   0.0387,   0.0272,   0.0054,  -0.0047,  -0.0023],
	[-0.0007,   -0.0030,  -0.0047,  -0.0025,  -0.0003,  -0.0025,  -0.0047,  -0.0030,  -0.0007],
	[-0.0001,   -0.0007,  -0.0023,  -0.0046,  -0.0057,  -0.0046,  -0.0023,  -0.0007,  -0.0001]
])

highpass = numpy.asarray([
	[0.0000,   0.0003,   0.0011,   0.0022,   0.0027,   0.0022,   0.0011,   0.0003,   0.0000],
	[0.0003,   0.0020,   0.0059,   0.0103,   0.0123,   0.0103,   0.0059,   0.0020,   0.0003],
	[0.0011,   0.0059,   0.0151,   0.0249,   0.0292,   0.0249,   0.0151,   0.0059,   0.0011],
	[0.0022,   0.0103,   0.0249,   0.0402,   0.0469,   0.0402,   0.0249,   0.0103,   0.0022],
	[0.0027,   0.0123,   0.0292,   0.0469,  -0.9455,   0.0469,   0.0292,   0.0123,   0.0027],
	[0.0022,   0.0103,   0.0249,   0.0402,   0.0469,   0.0402,   0.0249,   0.0103,   0.0022],
	[0.0011,   0.0059,   0.0151,   0.0249,   0.0292,   0.0249,   0.0151,   0.0059,   0.0011],
	[0.0003,   0.0020,   0.0059,   0.0103,   0.0123,   0.0103,   0.0059,   0.0020,   0.0003],
	[0.0000,   0.0003,   0.0011,   0.0022,   0.0027,   0.0022,   0.0011,   0.0003,   0.0000]
])

def getsize(img):
	h, w = img.shape[:2]
	return w, h

def build_laplacian(img, minsize=2, convolutionThreshold=500, dtype=numpy.float64):
	img = dtype(img)
	levels = []
	while (min(img.shape) > minsize):

		if (img.size < convolutionThreshold):
			convolutionFunction = scipy.signal.convolve2d
		else:
			convolutionFunction = scipy.signal.fftconvolve

		w, h = getsize(img)
		symmimg = symmetrical_boundary(img)
		
		hp_img = convolutionFunction(symmimg, highpass, mode='same')[h/2:-(h+1)/2,w/2:-(w+1)/2]
		lp_img = convolutionFunction(symmimg, lowpass, mode='same')[h/2:-(h+1)/2,w/2:-(w+1)/2]
		levels.append(hp_img)

		img = cv2.pyrDown(lp_img)

	levels.append(img)
	return levels

def collapse_laplacian(levels, convolutionThreshold=500):
	img = levels[-1]
	for ii in range(len(levels)-2,-1,-1):
		lev_img = levels[ii]

		img = cv2.pyrUp(img, dstsize=getsize(lev_img))

		if (img.size < convolutionThreshold):
			convolutionFunction = scipy.signal.convolve2d
		else:
			convolutionFunction = scipy.signal.fftconvolve

		w, h = getsize(img)
		symmimg = symmetrical_boundary(img)
		symmlev = symmetrical_boundary(lev_img)

		img = convolutionFunction(symmimg, lowpass, mode='same')[h/2:-(h+1)/2,w/2:-(w+1)/2]
		img += convolutionFunction(symmlev, highpass, mode='same')[h/2:-(h+1)/2,w/2:-(w+1)/2]
	return img