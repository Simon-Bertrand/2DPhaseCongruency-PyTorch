from .filtergrid import filtergrid


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
