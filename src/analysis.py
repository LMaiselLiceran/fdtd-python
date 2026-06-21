import numpy as np
from dataclasses import dataclass

@dataclass
class PeakProperties:
    position  : float   # physical position in metres
    amplitude : float   # signed amplitude
    index     : float   # sub-cell index (useful for debugging)

class FieldProbe:
    """
    Records the field value at a fixed spatial position over time.
    
    Parameters:
        grid     : Grid object
        position : physical position in metres
    """
    def __init__(self, grid, position):
        self.index  = round(position / grid.dx)
        self.record = []

    def measure(self, grid):
        """Record the current field value at the probe position"""
        self.record.append(grid.Ez[self.index])

    def get_series(self):
        """Return the recorded time series as a numpy array"""
        return np.array(self.record)

def find_peak(field_snapshot, dx, x_min=None, x_max=None):
    """
    Returns the physical position of the maximum of the field in a spatial window.
    Uses quadratic interpolation for sub-cell accuracy.

    Parameters:
        field_snapshot : 1D numpy array of field values
        dx             : spatial step in meters
        x_min (opt)    : left boundary of search window in meters
        x_max (opt)    : right boundary of search window in meters

    Returns:
        PeakProperties dataclass with position, amplitude, and index
    """
    i_min  = round(x_min / dx) if x_min is not None else 0
    i_max  = round(x_max / dx) if x_max is not None else len(field_snapshot)
    window = field_snapshot[i_min:i_max]

    i_peak = np.argmax(np.abs(window))

    # Sub-cell interpolation using quadratic fit around peak
    # (only if peak is not at the edge of the window)
    if 0 < i_peak < (len(window) - 1):
        y0 = np.abs(window[i_peak - 1])
        y1 = np.abs(window[i_peak])
        y2 = np.abs(window[i_peak + 1])
        # Quadratic interpolation: offset from peak cell
        offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        i_peak_interp = i_peak + offset
        amplitude = window[i_peak] + offset * (window[i_peak+1] - window[i_peak-1]) / 2
    else:
        i_peak_interp = i_peak
        amplitude = window[i_peak]

    index = i_min + i_peak_interp
    position = index * dx

    return PeakProperties(position=position, amplitude=amplitude, index=index)

def find_peak_in_window(field_history, dx, dt_snapshot, t_measure, x_min=None, x_max=None):
    """
    Finds the peak amplitude of a pulse in a spatial window at a specific measurement time

    Parameters:
        field_history : list of field snapshots saved at intervals dt_snapshot
        dx            : spatial step in meters
        dt_snapshot   : time interval between snapshots in seconds
        t_measure     : physical time at which to measure in seconds
        x_min         : left boundary of search window in metres (optional)
        x_max         : right boundary of search window in metres (optional)

    Returns:
        PeakProperties dataclass with position, amplitude, and index
    """

    i_frame = round(t_measure / dt_snapshot)

    if (i_frame > len(field_history)):
        raise ValueError(
            f"t_measure={t_measure} exceeds simulation duration. "
            f"Maximum frame index is {len(history)-1}, "
            f"corresponding to t={len(history)*dt_snapshot:.3e} s."
        )

    return find_peak(field_history[i_frame], dx, x_min, x_max)

def measure_wave_speed(field_history, dx, dt_snapshot, x_min=None, x_max=None):
    """
    Measures the numerical wave speed by tracking the peak position over time
    and fitting a line to position vs time

    Parameters:
        field_history : list of field snapshots saved at intervals dt_snapshot
        dx            : spatial step in meters
        dt_snapshot   : time interval between snapshots in seconds

    Returns:
        c_numerical : measured wave speed in m/s
        positions   : array of peak positions at each frame
        times       : array of times corresponding to each frame
    """
    positions = np.array([find_peak(snap, dx, x_min, x_max).position for snap in field_history])
    times     = np.arange(len(field_history)) * dt_snapshot

    # Linear fit with covariance: position = c * time + offset
    coeffs, cov = np.polyfit(times, positions, deg=1, cov=True)
    c_numerical = coeffs[0]
    c_error     = np.sqrt(cov[0, 0])

    return c_numerical, c_error, positions, times

def compute_transfer_function(incident_series, transmitted_series, dt_sample, window=True):
    """
    Computes the frequency-domain transfer function H(f) = E_trans(f) / E_inc(f).

    Parameters:
        incident_series    : 1D array of incident field values at a fixed point
        transmitted_series : 1D array of transmitted field values at a fixed point
        window             : if True, apply a Hanning window before FFT to reduce
                             spectral leakage (default: True)

    Returns:
        freqs   : array of frequencies in Hz
        H_amp   : amplitude of transfer function |H(f)|
        H_phase : phase of transfer function angle(H(f)) in radians
    """

    n = len(incident_series)

    if (len(transmitted_series) != n):
        raise ValueError("incident_series and transmitted_series must have the same length.")

    if window:
        w = np.hanning(n)
        incident_series    = incident_series    * w
        transmitted_series = transmitted_series * w

    # FFT both series
    E_inc   = np.fft.rfft(incident_series)
    E_trans = np.fft.rfft(transmitted_series)
    freqs   = np.fft.rfftfreq(n, d=dt_sample)

    # Avoid division by near-zero
    threshold = 1e-14 * np.max(np.abs(E_inc))
    mask      = np.abs(E_inc) > threshold

    H = np.zeros_like(E_inc, dtype=complex)
    H[mask] = E_trans[mask] / E_inc[mask]

    return freqs, np.abs(H), np.angle(H), E_inc, E_trans