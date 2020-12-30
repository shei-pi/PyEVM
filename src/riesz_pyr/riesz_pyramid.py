"""
riesz_pyramid.py
Conversion between Riesz and Laplacian image pyramids
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

import numpy, math
import scipy, scipy.signal

#riesz_band_filter = numpy.asarray([[-0.5, 0, 0.5]])
#riesz_band_filter = numpy.asarray([[-0.2,-0.48, 0, 0.48,0.2]])
riesz_band_filter = numpy.asarray([[-0.12,0,0.12],[-0.34, 0, 0.34],[-0.12,0,0.12]])

def laplacian_to_riesz(pyr):
	newpyr = {'I':pyr[:-1], 'R1':[], 'R2':[]}
	for ii in range(len(pyr) - 1):
		newpyr['R1'].append( scipy.signal.convolve2d(pyr[ii], riesz_band_filter, mode='same', boundary='symm') )
		newpyr['R2'].append( scipy.signal.convolve2d(pyr[ii], riesz_band_filter.T, mode='same', boundary='symm') )
	newpyr['base'] = pyr[-1]
	return newpyr

def riesz_to_spherical(pyr):
	newpyr = {'A':[],'theta':[],'phi':[],'Q':[],'base':pyr['base']}
	for ii in range(len(pyr['I']) ):
		I = pyr['I'][ii]
		R1 = pyr['R1'][ii]
		R2 = pyr['R2'][ii]
		A = numpy.sqrt(I*I + R1*R1 + R2*R2)
		theta = numpy.arctan2(R2,R1)
		Q = R1 * numpy.cos(theta) + R2 * numpy.sin(theta)
		phi = numpy.arctan2(Q,I)

		newpyr['A'].append( A )
		newpyr['theta'].append( theta )
		newpyr['phi'].append( phi )
		newpyr['Q'].append( Q )
	return newpyr


def riesz_spherical_to_laplacian(pyr):
	newpyr = []
	for ii in range(len(pyr['A'])):
		newpyr.append( pyr['A'][ii] * numpy.cos( pyr['phi'][ii] ) )
	newpyr.append(pyr['base'])
	return newpyr