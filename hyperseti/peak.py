import cupy as cp
from cupyx.scipy import ndimage as ndi
import time

def prominent_peaks(img, min_xdistance=1, min_ydistance=1, threshold=None, num_peaks=cp.inf):
    """Return peaks with non-maximum suppression.
    Identifies most prominent features separated by certain distances.
    Non-maximum suppression with different sizes is applied separately
    in the first and second dimension of the image to identify peaks.
    
    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    min_xdistance : int
        Minimum distance separating features in the x dimension.
    min_ydistance : int
        Minimum distance separating features in the y dimension.
    threshold : float
        Minimum intensity of peaks. Default is `0.5 * max(image)`.
    num_peaks : int
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.
        
    Returns
    -------
    intensity, xcoords, ycoords : tuple of array
        Peak intensity values, x and y indices.
        
    Notes
    -----
    Modified from https://github.com/mritools/cupyimg _prominent_peaks method
    """

    #img = image.copy()
    rows, cols = img.shape

    if threshold is None:
        threshold = 0.5 * cp.max(img)

    ycoords_size = 2 * min_ydistance + 1
    xcoords_size = 2 * min_xdistance + 1
    
    t0 = time.time()
    img_max = ndi.maximum_filter(
        img, size=(ycoords_size, xcoords_size), mode="constant", cval=0
    )
    te = (time.time() - t0) * 1e3
    print(f"Maxfilter: {te:2.2f} ms")

    t0 = time.time()
    mask = img == img_max
    img *= mask
    mask = img > threshold
    te = (time.time() - t0) * 1e3
    print(f"bitbash: {te:2.2f} ms")

    t0 = time.time()
    # Find array (x,y) indexes corresponding to max pixels
    peak_idxs = cp.argwhere(mask)
    # Find corresponding maximum values
    peak_vals = img[peak_idxs[:, 0], peak_idxs[:, 1]]
    # Sort peak values low to high
    ## Sort the list of peaks by intensity, not left-right, so larger peaks
    ## in Hough space cannot be arbitrarily suppressed by smaller neighbors
    val_sort_idx = cp.argsort(peak_vals)[::-1]
    # Return (x,y) coordinates corresponding to sorted max pixels
    coords = peak_idxs[val_sort_idx]
    te = (time.time() - t0) * 1e3
    print(f"coord search: {te:2.2f} ms")
    
    t0 = time.time()
    img_peaks = []
    ycoords_peaks = []
    xcoords_peaks = []

    # relative coordinate grid for local neighbourhood suppression
    ycoords_ext, xcoords_ext = cp.mgrid[
        -min_ydistance : min_ydistance + 1, -min_xdistance : min_xdistance + 1
    ]

    for ycoords_idx, xcoords_idx in coords:
        accum = img_max[ycoords_idx, xcoords_idx]
        if accum > threshold:
            # absolute coordinate grid for local neighbourhood suppression
            ycoords_nh = ycoords_idx + ycoords_ext
            xcoords_nh = xcoords_idx + xcoords_ext

            # no reflection for distance neighbourhood
            ycoords_in = cp.logical_and(ycoords_nh > 0, ycoords_nh < rows)
            ycoords_nh = ycoords_nh[ycoords_in]
            xcoords_nh = xcoords_nh[ycoords_in]

            # reflect xcoords and assume xcoords are continuous,
            # e.g. for angles:
            # (..., 88, 89, -90, -89, ..., 89, -90, -89, ...)
            xcoords_low = xcoords_nh < 0
            ycoords_nh[xcoords_low] = rows - ycoords_nh[xcoords_low]
            xcoords_nh[xcoords_low] += cols
            xcoords_high = xcoords_nh >= cols
            ycoords_nh[xcoords_high] = rows - ycoords_nh[xcoords_high]
            xcoords_nh[xcoords_high] -= cols

            # suppress neighbourhood
            img_max[ycoords_nh, xcoords_nh] = 0

            # add current feature to peaks
            img_peaks.append(accum)
            ycoords_peaks.append(ycoords_idx)
            xcoords_peaks.append(xcoords_idx)

    img_peaks = cp.array(img_peaks)
    ycoords_peaks = cp.array(ycoords_peaks)
    xcoords_peaks = cp.array(xcoords_peaks)
    te = (time.time() - t0) * 1e3
    print(f"crazyloop: {te:2.2f} ms")
    
    if num_peaks < len(img_peaks):
        idx_maxsort = cp.argsort(img_peaks)[::-1][:num_peaks]
        img_peaks = img_peaks[idx_maxsort]
        ycoords_peaks = ycoords_peaks[idx_maxsort]
        xcoords_peaks = xcoords_peaks[idx_maxsort]

    return img_peaks, xcoords_peaks, ycoords_peaks