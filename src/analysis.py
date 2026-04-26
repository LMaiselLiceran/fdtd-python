import numpy as np

def find_peak_position(field_snapshot, dx, x_min=None, x_max=None):
    """
    Returns the physical position of the maximum of the field snapshot

    Parameters:
        field_snapshot : 1D numpy array of field values
        dx             : spatial step in meters
        x_min (opt)    : left boundary of search window in meters
        x_max (opt)    : right boundary of search window in meters
    """
    i_min = round(x_min / dx) if x_min is not None else 0
    i_max = round(x_max / dx) if x_max is not None else len(field_snapshot)

    window  = field_snapshot[i_min:i_max]
    i_peak = np.argmax(np.abs(window))

    # Sub-cell interpolation using quadratic fit around peak
    # (only if peak is not at the edge of the window)
    if 0 < i_peak < len(window) - 1:
        y0 = np.abs(window[i_peak - 1])
        y1 = np.abs(window[i_peak])
        y2 = np.abs(window[i_peak + 1])
        # Quadratic interpolation: offset from peak cell
        offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        i_peak_interp = i_peak + offset
    else:
        i_peak_interp = i_peak

    return (i_min + i_peak_interp) * dx

def measure_wave_speed(field_history, dx, dt_snapshot, x_min=None, x_max=None):
    """
    Measures the numerical wave speed by tracking the peak position over time
    and fitting a line to position vs time

    Parameters
        history      : list of field snapshots saved at intervals dt_snapshot
        dx           : spatial step in meters
        dt_snapshot  : time interval between snapshots in seconds

    Returns:
        c_numerical : measured wave speed in m/s
        positions   : array of peak positions at each frame
        times       : array of times corresponding to each frame
    """
    positions = np.array([find_peak_position(snap, dx, x_min, x_max) for snap in field_history])
    times     = np.arange(len(field_history)) * dt_snapshot

    # Linear fit with covariance: position = c * time + offset
    coeffs, cov = np.polyfit(times, positions, deg=1, cov=True)
    c_numerical = coeffs[0]
    c_error     = np.sqrt(cov[0, 0])

    return c_numerical, c_error, positions, times