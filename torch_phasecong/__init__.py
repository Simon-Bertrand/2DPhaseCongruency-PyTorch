#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
This code is a rewrite for PyTorch that Simon BERTRAND made for the computation of phase congruency using monogenic filters, which has been originally written by Peter Kovesi in MatLab :
Check the reference here : https://www.peterkovesi.com/matlabfns/PhaseCongruency/phasecongmono.m
His website is : https://www.peterkovesi.com
He gave some explanations about phase congruency here : https://peterkovesi.com/projects/phasecongruency/index.html
Please, quote its work if you use these functions.

For fast explanations about phase congruency, you can check there : 
https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT7/node2.html
Phase congruency is very useful because of its edges and corners detection not being based on pixels intensities, which can be interesting for the multimodal data fusion in machine learning.

"""

import torch, math
import warnings


def filtergrid(rows, cols):
    """FILTERGRID Generates grid for constructing frequency domain filters
    Usage:  [radius, u1, u2] = filtergrid(rows, cols)
            [radius, u1, u2] = filtergrid([rows, cols])

    Used by PHASECONGMONO, PHASECONG3 etc etc

    See also: WAVENUMBERGRID
    Copyright (c) 1996-2017 Peter Kovesi
    Centre for Exploration Targeting
    The University of Western Australia
    peter.kovesi at uwa edu au

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    The Software is provided "as is", without warranty of any kind.

    May 2013
    September 2017 Correction to setting up matrices of frequency values for
                    odd sized images.
    Args:
        rows (int):  Size of image/filter
        cols (int): Size of image/filter

    Returns:
        radius (torch.Tensor) : Grid of size [rows cols] containing normalised radius values from 0 to 0.5. Grid is quadrant shifted so that 0 frequency is at radius(1,1)
        u1, u2 (torch.Tensor) : Grids containing normalised frequency values ranging from -0.5 to 0.5 in x and y directions respectively. u1 and u2 are quadrant shifted.
    """

    # Set up X and Y spatial frequency matrices, u1 and u2 The following code
    # adjusts things appropriately for odd and even values of rows and columns
    # so that the 0 frequency point is placed appropriately.  See
    # https://blogs.uoregon.edu/seis/wiki/unpacking-the-matlab-fft/
    if cols % 2:
        u1range = torch.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / cols
    else:
        u1range = torch.arange(-cols / 2, cols / 2 - 1 + 1) / cols

    if rows % 2:
        u2range = torch.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / rows
    else:
        u2range = torch.arange(-rows / 2, rows / 2 - 1 + 1) / rows

    u1, u2 = torch.meshgrid(u1range, u2range, indexing="xy")

    # Quadrant shift so that filters are constructed with 0 frequency at the corners
    u1 = torch.fft.ifftshift(u1)
    u2 = torch.fft.ifftshift(u2)

    # Construct spatial frequency values in terms of normalized radius from the center
    radius = torch.sqrt(u1**2 + u2**2)

    return radius, u1, u2


def lowpassfilter(sze, cutoff, n):
    """LOWPASSFILTER - Constructs a low-pass butterworth filter.
    usage: f = lowpassfilter(sze, cutoff, n)

    where: sze    is a two element vector specifying the size of filter
                to construct [rows cols].
            cutoff is the cutoff frequency of the filter 0 - 0.5
            n      is the order of the filter, the higher n is the sharper
                the transition is. (n must be an integer >= 1).
                Note that n is doubled so that it is always an even integer.

                        1
        f =    --------------------
                                2n
                1.0 + (w/cutoff)

    The frequency origin of the returned filter is at the corners.

    See also: HIGHPASSFILTER, HIGHBOOSTFILTER, BANDPASSFILTER, FILTERGRID

    Copyright (c) 1999 Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    http://www.csse.uwa.edu.au/

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    The Software is provided "as is", without warranty of any kind.
    October 1999
    August  2005 - Fixed up frequency ranges for odd and even sized filters
                    (previous code was a bit approximate)
    September 2017 - Changed to use filtergrid()

    Args:
        sze (tuple or list): is a two element vector specifying the size of filter to construct [rows, cols].
        cutoff (float): is the cutoff frequency of the filter 0 - 0.5
        n (int): is the order of the filter, the higher n is the sharper the transition is. (n must be an integer >= 1).
                Note that n is doubled so that it is always an even integer.

    Raises:
        ValueError: Cutoff frequency must be between 0 and 0.5
        ValueError: n must be an integer >= 1

    Returns:
        f (torch.Tensor) : The frequency origin of the returned filter is at the corners.
    """
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError("Cutoff frequency must be between 0 and 0.5")

    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")

    if isinstance(sze, int):
        rows, cols = sze, sze
    else:
        rows, cols = sze[0], sze[1]
    radius, _, _ = filtergrid(rows, cols)
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * n))


def perfft2(im):
    """PERFFT2  2D Fourier transform of Moisan's periodic image component

    Moisan's "Periodic plus Smooth Image Decomposition" decomposes an image
    into two components
           im = p + s
    where s is the 'smooth' component with mean 0 and p is the 'periodic'
    component which has no sharp discontinuities when one moves cyclically across
    the image boundaries.

    This wonderful decomposition is very useful when one wants to obtain an FFT of
    an image with minimal artifacts introduced from the boundary discontinuities.
    The image p gathers most of the image information but avoids periodization
    artifacts.

    The typical use of this function is to obtain a 'periodic only' fft of an
    image
      >>  P = perfft2(im);

    Displaying the amplitude spectrum of P will yield a clean spectrum without the
    typical vertical-horizorescale(ntal 'cross' arising from the image boundaries that you
    would normally see.

    Note if you are using the function to perform filtering in the frequency
    domain you may want to retain s (the smooth component in the spatial domain)
    and add it back to the filtered result at the end.

    The computational cost of obtaining the 'periodic only' FFT involves taking an
    additional FFT.


    Reference:
    This code is adapted from Lionel Moisan's Scilab function 'perdecomp.sci'
    "Periodic plus Smooth Image Decomposition" 07/2012 available at

      http://www.mi.parisdescartes.fr/~moisan/p+s

    Paper:
    L. Moisan, "Periodic plus Smooth Image Decomposition", Journal of
    Mathematical Imaging and Vision, vol 39:2, pp. 161-179, 2011.
    Peter Kovesi
    Centre for Exploration Targeting
    The University of Western Australia
    peter.kovesi at uwa edu au
    September 2012

       Args:
           im (torch.Tensor): Image to be transformed

       Returns:
           P (torch.Tensor) : 2D fft of periodic image component
           S (torch.Tensor) : 2D fft of smooth component
    """

    if im.dtype != torch.float64:
        im = im.double()

    rows, cols = im.shape
    s = torch.zeros_like(im)
    s[0, :] = im[0, :] - im[-1, :]
    s[-1, :] = -s[0, :]
    s[:, 0] = s[:, 0] + im[:, 0] - im[:, -1]
    s[:, -1] = s[:, -1] - im[:, 0] + im[:, -1]

    cx, cy = torch.meshgrid(
        2 * torch.pi * torch.arange(cols, dtype=torch.float64) / cols,
        2 * torch.pi * torch.arange(rows, dtype=torch.float64) / rows,
        indexing="xy",
    )

    S = torch.fft.fft2(s) / (2 * (2 - torch.cos(cx) - torch.cos(cy)))
    S[0, 0] = 0  # Remove DC component <=> Set the mean of image to zero

    return torch.fft.fft2(im) - S, S


def phasecong(
    im,
    nscale=4,
    minWaveLength=3,
    mult=2.1,
    sigmaOnf=0.55,
    k=3.0,
    noiseMethod=-1,
    cutOff=0.5,
    g=10,
    deviationGain=1.5,
):
    """PHASECONGMONO - phase congruency of an image using monogenic filters

    This code is considerably faster than PHASECONG3 but you may prefer the
    output from PHASECONG3's oriented filters.

    There are potentially many arguments, here is the full usage:

      [PC or ft T] =  ...
                   phasecongmono(im, nscale, minWaveLength, mult, ...
                            sigmaOnf, k, cutOff, g, deviationGain, noiseMethod)

    However, apart from the image, all parameters have defaults and the
    usage can be as simple as:

       phaseCong = phasecongmono(im);

    Notes on specifying parameters:

    The parameters can be specified as a full list eg.
     >> PC = phasecongmono(im, 5, 3, 2.5, 0.55, 2.0);

    or as a partial list with unspecified parameters taking on default values
     >> PC = phasecongmono(im, 5, 3);

    or as a partial list of parameters followed by some parameters specified via a
    keyword-value pair, remaining parameters are set to defaults, for example:
     >> PC = phasecongmono(im, 5, 3, 'k', 2.5);

    The convolutions are done via the FFT.  Many of the parameters relate to the
    specification of the filters in the frequency plane.  The values do not seem
    to be very critical and the defaults are usually fine.  You may want to
    experiment with the values of 'nscales' and 'k', the noise compensation
    factor.

    Typical sequence of operations to obtain an edge image:

     >> [PC, or] = phasecongmono(imread('lena.tif'));
     >> nm = nonmaxsup(PC, or, 1.5);   % nonmaxima suppression
     >> bw = hysthresh(nm, 0.1, 0.3);  % hysteresis thresholding 0.1 - 0.3
     >> show(bw)

    Notes on filter settings to obtain even coverage of the spectrum
    sigmaOnf       .85   mult 1.3
    sigmaOnf       .75   mult 1.6     (filter bandwidth ~1 octave)
    sigmaOnf       .65   mult 2.1
    sigmaOnf       .55   mult 3       (filter bandwidth ~2 octaves)

    Note that better results are achieved using the large bandwidth filters.
    I generally use a sigmaOnf value of 0.55 or even smaller.

    See Also:  PHASECONG, PHASECONG3, PHASESYMMONO, GABORCONVOLVE,
    PLOTGABORFILTERS, FILTERGRID
    References:

        Peter Kovesi, "Image Features From Phase Congruency". Videre: A
        Journal of Computer Vision Research. MIT Press. Volume 1, Number 3,
        Summer 1999 http://mitpress.mit.edu/e-journals/Videre/001/v13.html

        Michael Felsberg and Gerald Sommer, "A New Extension of Linear Signal
        Processing for Estimating Local Properties and Detecting Features". DAGM
        Symposium 2000, Kiel

        Michael Felsberg and Gerald Sommer. "The Monogenic Signal" IEEE
        Transactions on Signal Processing, 49(12):3136-3144, December 2001

        Peter Kovesi, "Phase Congruency Detects Corners and Edges". Proceedings
        DICTA 2003, Sydney Dec 10-12
    August 2008    Initial version developed from phasesymmono and phasecong2
                   where local phase information is calculated via Monogenic
                   filters. Simplification of noise compensation to speedup
                   execution. Options to calculate noise statistics via median
                   or mode of smallest filter response.
    April 2009     Return of T for 'instrumentation' of noise
                   detection/compensation. Option to use a fixed threshold.
                   Frequency width measure slightly improved.
    June 2009      20% Speed improvement through calculating phase deviation via
                   acos() rather than computing cos(theta)-|sin(theta)| via dot
                   and cross products.  Phase congruency is formed properly in
                   2D rather than as multiple 1D computations as in
                   phasecong3. Also, much smaller memory footprint.
    May 2013       Some tidying up, corrections to defualt parameters, changes
                   to reflect my latest thinking of the final phase congruency
                   calculation and addition of phase deviation gain parameter to
                   sharpen up the output. Use of periodic fft.
    Copyright (c) 1996-2013 Peter Kovesi
    Centre for Exploration Targeting
    The University of Western Australia
    peter.kovesi at uwa edu au

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    The Software is provided "as is", without warranty of any kind.

    Args:
        im (torch.Tensor): Image to compute the phase congruency
        nscale (int, optional): Number of wavelet scales, try values 3-6
                               A lower value will reveal more fine scale
                               features. A larger value will highlight 'major'
                               features. Defaults to 4.
        minWaveLength (int, optional): Wavelength of smallest scale filter. Defaults to 3.
        mult (float, optional): Scaling factor between successive filters. Defaults to 2.1.
        sigmaOnf (float, optional): Ratio of the standard deviation of the Gaussian
                               describing the log Gabor filter's transfer function
                               in the frequency domain to the filter center frequency. Defaults to 0.55.
        k (float, optional): No of standard deviations of the noise energy beyond the mean at which we set the noise threshold point. You may want to vary this up to a value of 10 or 20 for noisy images. Defaults to 3.0.
        noiseMethod (int, optional): Parameter specifies method used to determine
                               noise statistics.
                                 -1 use median of smallest scale filter responses
                                 -2 use mode of smallest scale filter responses
                                  0+ use noiseMethod value as the fixed noise threshold. A value of 0 will turn off all noise compensation. Defaults to -1.
        cutOff (float, optional): The fractional measure of frequency spread. Defaults to 0.5.
        g (int, optional): Controls the sharpness of the transition in the sigmoid function used to weight phase congruency for frequency spread. Defaults to 10.
        deviationGain (float, optional): Amplification to apply to the calculated phase
                               deviation result. Increasing this sharpens the
                               edge responses, but can also attenuate their
                               magnitude if the gain is too large.  Sensible
                               values to use lie in the range 1-2. Defaults to 1.5.

    Returns:
        PC (torch.Tensor): Phase congruency indicating edge significance
        ori (torch.Tensor) : Orientation image in integer degrees 0-180,  positive anticlockwise. 0 corresponds to a vertical edge, 90 is horizontal.
        ft (torch.Tensor) : Local weighted mean phase angle at every point in the image.  A value of pi/2 corresponds to a bright line, 0 corresponds to a step and -pi/2 is a dark line.
        T (torch.Tensor) : Calculated noise threshold (can be useful for diagnosing noise characteristics of images).  Once you know this you can then specify fixed thresholds and save some computation time.
    """
    phasecong_checkargs(
        im,
        nscale,
        minWaveLength,
        mult,
        sigmaOnf,
        k,
        noiseMethod,
        cutOff,
        g,
        deviationGain,
    )

    epsilon = 0.0001  # Used to prevent division by zero.

    rows, cols = im.shape
    IM, _ = perfft2(im)  # You need to implement perfft2 based on your needs

    sumAn = torch.zeros(
        rows, cols
    )  # Matrix for accumulating filter response amplitude values.
    sumf = torch.zeros(rows, cols)
    sumh1 = torch.zeros(rows, cols)
    sumh2 = torch.zeros(rows, cols)

    radius, u1, u2 = filtergrid(rows, cols)
    radius[0, 0] = 1

    H = (1j * u1 - u2) / radius

    lp = lowpassfilter([rows, cols], 0.45, 15)  # Radius 0.45, 'sharpness' 15

    for s in range(1, nscale + 1):
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength  # Centre frequency of filter.
        logGabor = torch.exp(
            -(torch.square((torch.log(radius / fo)))) / (2 * math.log(sigmaOnf) ** 2)
        )
        logGabor = logGabor * lp
        logGabor[0, 0] = 0
        IMF = IM * logGabor
        f = torch.fft.ifft2(IMF).real
        h = torch.fft.ifft2(IMF * H)
        h1 = h.real
        h2 = h.imag
        An = torch.sqrt(f**2 + h1**2 + h2**2)
        sumAn += An
        sumf += f
        sumh1 += h1
        sumh2 += h2

        if s == 1:
            if noiseMethod == -1:
                tau = torch.median(sumAn) / math.sqrt(math.log(4))
            elif noiseMethod == -2:
                tau = rayleighmode(sumAn)
            maxAn = An
        else:
            maxAn = torch.max(maxAn, An)

    width = (sumAn / (maxAn + epsilon) - 1) / (nscale - 1)
    weight = 1.0 / (1 + torch.exp((cutOff - width) * g))

    if noiseMethod >= 0:
        T = noiseMethod
    else:
        totalTau = tau * (1 - (1 / mult) ** nscale) / (1 - (1 / mult))
        EstNoiseEnergyMean = totalTau * math.sqrt(torch.pi / 2)
        EstNoiseEnergySigma = totalTau * math.sqrt((4 - torch.pi) / 2)
        T = EstNoiseEnergyMean + k * EstNoiseEnergySigma

    ori = torch.atan(-sumh2 / sumh1)
    ori[ori < 0] += torch.pi
    ori = torch.fix(ori / torch.pi * 180)

    ft = torch.atan2(sumf, torch.sqrt(sumh1**2 + sumh2**2))
    energy = torch.sqrt(sumf**2 + sumh1**2 + sumh2**2)
    PC = (
        weight
        * ((1 - deviationGain * torch.acos(energy / (sumAn + epsilon))).clamp(min=0))
        * (energy - T).clamp(min=0)
        / (energy + epsilon)
    )

    return PC, ori, ft, T


def rayleighmode(data, nbins=50):
    """RAYLEIGHMODE
    Computes mode of a vector/matrix of data that is assumed to come from a
    Rayleigh distribution.
    Usage:  rmode = rayleighmode(data, nbins)
    Arguments:  data  - data assumed to come from a Rayleigh distribution
                nbins - Optional number of bins to use when forming histogram
                        of the data to determine the mode.
    Mode is computed by forming a histogram of the data over 50 bins and then
    finding the maximum value in the histogram.  Mean and standard deviation
    can then be calculated from the mode as they are related by fixed
    constants.
    mean = mode * sqrt(pi/2)
    std dev = mode * sqrt((4-pi)/2)

    See
    http://mathworld.wolfram.com/RayleighDistribution.html
    http://en.wikipedia.org/wiki/Rayleigh_distribution

    Args:
        data (torch.Tensor): data assumed to come from a Rayleigh distribution
        nbins (int, optional): Optional number of bins to use when forming histogram of the data to determine the mode. Defaults to 50.

    Returns:
        rmode (float): The Rayleigh mode value
    """
    mx = torch.max(data)
    edges = torch.linspace(0, mx, nbins + 1)
    n, _ = torch.histogram(data, bins=edges)
    ind = torch.argmax(
        torch.tensor(n)
    )  # Find maximum and index of maximum in histogram

    rmode = (edges[ind] + edges[ind + 1]) / 2

    return rmode


def phasecong_checkargs(
    im, nscale, minWaveLength, mult, sigmaOnf, k, noiseMethod, cutOff, g, deviationGain
):
    if not isinstance(im, torch.Tensor):
        raise ValueError("Image must be a torch.Tensor")

    if len(im.shape) == 3:
        warnings.warn("Colour image supplied: converting image to greyscale...")
        im = im.mean(dim=-1)

    if im.dtype != torch.float64:
        im = im.double()

    if nscale < 1:
        raise ValueError("nscale must be an integer >= 1")

    if minWaveLength < 2:
        raise ValueError("It makes little sense to have a wavelength < 2")