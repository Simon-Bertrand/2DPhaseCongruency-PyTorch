import torch


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
