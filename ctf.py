import numpy as np
import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ctf_freqs(shape, d=1.0, full=True): #generates frq grds (s and a) for CTF eval
    """
    :param shape: Shape tuple.
    :param d: Frequency spacing in inverse Å (1 / pixel size).
    :param full: When false, return only unique Fourier half-space for real data.
    """
    if full:
        xfrq = torch.from_numpy(np.fft.fftfreq(shape[1])) #compute freq for full fourier space
    else:
        xfrq = torch.from_numpy(np.fft.rfftfreq(shape[1])) #compute freq for half fourier space
    x, y = torch.meshgrid(xfrq, torch.from_numpy(np.fft.fftfreq(shape[0])))
    rho = torch.sqrt(x ** 2. + y ** 2.)
    a = torch.atan2(y, x)
    s = (rho * d)
    return s, a

def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0): #evaluaate ctf
    """
    :param s: Precomputed frequency grid for CTF evaluation.
    :param a: Precomputed frequency grid angles.
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = torch.deg2rad(angast)
    kv =kv * 1e3
    cs = cs* 1e7
    lamb = 12.2643247 / torch.sqrt(kv * (1. + kv * 0.978466e-6)) #gives batch size 500
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2. * lamb
    k2 = np.pi / 2. * cs * lamb ** 3.
    k3 = np.sqrt(1. - ac ** 2.)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s ** 2.
    s_4 = s_2 ** 2.
    dZ = def_avg[:, None, None] + def_dev[:, None, None] * (torch.cos(2. * (a - angast[:, None, None])))
    gamma = (k1[:, None, None] * dZ * s_2) + (k2[:, None, None] * s_4) - k5
    ctf = -(k3 * torch.sin(gamma) - ac * torch.cos(gamma)) #this is the formula of the ctf
    if bf != 0:  # Enforce envelope.
        ctf *= torch.exp(-k4 * s_2)
    return ctf

def computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr, img_shape, batch_size, applyCTF=True):
    if applyCTF:
        s, a = ctf_freqs([img_shape[0], img_shape[0]], 1 / sr)
        s, a = torch.tile(s[None, :, :], [batch_size, 1, 1]), torch.tile(a[None, :, :], [batch_size, 1, 1])
        ctf = eval_ctf(s, a, defocusU, defocusV, angast=defocusAngle, cs=cs, kv=kv)
        ctf = torch.fft.fftshift(ctf[:, :, :img_shape[1]])
        return ctf
    else:
        return torch.ones([batch_size, img_shape[0], img_shape[1]], dtype=torch.float32)