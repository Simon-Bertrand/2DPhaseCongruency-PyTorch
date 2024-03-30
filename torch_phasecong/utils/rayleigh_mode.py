import torch


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
    ind = torch.argmax(torch.tensor(n))  # Find maximum and index of maximum in histogram

    rmode = (edges[ind] + edges[ind + 1]) / 2

    return rmode
