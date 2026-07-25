"""
reference_impl.py — Pipeline de librosa beat detection RE-IMPLEMENTADO DESDE CERO.

NO llama a librosa.stft / melspectrogram / tempogram / beat_track.
Solo usa numpy + scipy.fft (que es lo que el C++ tendra que reproducir con KissFFT).

Objetivo: DEMOSTRAR que las formulas corregidas (ver SECCION8_ANALISIS_CRITICO.md)
reproducen el golden master generado por librosa. Si esto cuadra, el C++ es mera
traducción 1:1.

Cada funcion lleva un comentario indicando la correccion (C1..C4) o añadido (A1..A5)
que aplica respecto al doc LIBROSA_BPM_PORT_CPP.md original.
"""
from __future__ import annotations
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len


# =============================================================================
#  Constantes
# =============================================================================
SR = 22050
HOP = 512
N_FFT = 2048
N_MELS = 128
FMAX = SR / 2.0
AMIN = 1e-10
TOP_DB = 80.0
START_BPM = 120.0
STD_BPM = 1.0
AC_SIZE = 8.0
MAX_TEMPO = 320.0
TIGHTNESS = 100.0
WIN_LENGTH = int(np.floor(AC_SIZE * SR / HOP))   # 344


# =============================================================================
#  Etapa 1.0 — ventanas y mel  (fórmulas CORREGIDAS)
# =============================================================================
def hann_periodic(N: int) -> np.ndarray:
    """C1: Hann periodica = 0.5 - 0.5*cos(2*pi*n/N)   (denominador N, no N+1 ni N-1)"""
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N)


def hanning5() -> np.ndarray:
    """C3: np.hanning(5) = [0, 0.5, 1.0, 0.5, 0]"""
    return np.hanning(5)


def hz_to_mel(f):
    """Slaney: f<1000 -> f/66.6667 ; f>=1000 -> 15 + ln(f/1000)/0.06875178"""
    f = np.asarray(f, dtype=np.float64)
    return np.where(f < 1000.0, f / 66.66666666666667,
                    15.0 + np.log(f / 1000.0) / 0.06875177742094912)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    return np.where(m < 15.0, m * 66.66666666666667,
                    1000.0 * np.exp(0.06875177742094912 * (m - 15.0)))


def build_mel_filterbank() -> np.ndarray:
    """C4: fórmula del codigo REAL de librosa, no la del doc (que daba suma 0).
       lower=(f-mel_f[i])/(mel_f[i+1]-mel_f[i])  (ascendente)
       upper=(mel_f[i+2]-f)/(mel_f[i+2]-mel_f[i+1])  (descendente)
       H[i,k]=max(0,min(lower,upper)) * 2/(mel_f[i+2]-mel_f[i])   (Slaney enorm)"""
    fftfreqs = np.arange(1 + N_FFT // 2) * SR / N_FFT          # 1025 frecuencias
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(FMAX), N_MELS + 2)
    mel_f = mel_to_hz(mel_points)                              # 130 Hz centers
    fdiff = np.diff(mel_f)                                     # 129 diffs
    weights = np.zeros((N_MELS, 1 + N_FFT // 2), dtype=np.float64)
    for i in range(N_MELS):
        lower = (fftfreqs - mel_f[i]) / fdiff[i]               # sube de 0->1
        upper = (mel_f[i + 2] - fftfreqs) / fdiff[i + 1]       # baja de 1->0
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
        weights[i] *= 2.0 / (mel_f[i + 2] - mel_f[i])          # enorm Slaney
    return weights


# =============================================================================
#  Etapa 1.1 — STFT (centering + RFFT + |.|^2)
# =============================================================================
def stft_power(y: np.ndarray) -> np.ndarray:
    """C2: M = 1 + N//HOP. Pad 1024 ceros cada lado. RFFT de 2048, |.|^2."""
    N = len(y)
    M = 1 + N // HOP                       # C2 corregido
    ypad = np.pad(y, (N_FFT // 2, N_FFT // 2))     # 1024 ceros a cada lado
    w = hann_periodic(N_FFT)
    S = np.empty((1 + N_FFT // 2, M), dtype=np.float64)
    for m in range(M):
        frame = ypad[m * HOP:m * HOP + N_FFT] * w
        X = rfft(frame, n=N_FFT)            # longitud 1025
        S[:, m] = (X.real ** 2 + X.imag ** 2)
    return S


# =============================================================================
#  Etapa 1.3/1.4 — power_to_db  (A1: top_db=80 ACTIVO)
# =============================================================================
def power_to_db(S: np.ndarray, amin: float = AMIN, top_db: float = TOP_DB) -> np.ndarray:
    """A1: clamp del rango dinamico a top_db=80 dB por debajo del pico."""
    magnitude = np.asarray(S, dtype=np.float64)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, 1.0))   # ref=1.0
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


# =============================================================================
#  Etapa 1.5 — onset strength  (median, lag=1, pad 3)
# =============================================================================
def onset_strength(S_dB: np.ndarray) -> np.ndarray:
    """flux = max(0, S_dB[:,m] - S_dB[:,m-1]); oenv_raw = median_128(flux[:,m]).
       Padding de 3 ceros al inicio (lag + n_fft/(2*hop) = 1+2). Trim a M."""
    M = S_dB.shape[1]
    flux = np.maximum(0.0, S_dB[:, 1:] - S_dB[:, :-1])      # (128, M-1)
    oenv_raw = np.median(flux, axis=0)                      # (M-1,)  median de 128
    pad = 1 + N_FFT // (2 * HOP)                            # 3
    oenv = np.concatenate([np.zeros(pad), oenv_raw])        # (M+2,)
    oenv = oenv[:M]                                         # trim a M
    return oenv


# =============================================================================
#  Etapa 2 — tempogram + tempo  (A3: argmax(bpms<320); A5: log1p(1e6*tg))
# =============================================================================
def autocorrelate_bounded(x: np.ndarray, max_size: int) -> np.ndarray:
    """AC por FFT (Wiener-Khinchin). n_pad = next_fast_len(2*len-1, real)."""
    n_pad = next_fast_len(2 * len(x) - 1, real=True)
    powspec = np.abs(rfft(x, n=n_pad)) ** 2
    ac = irfft(powspec, n=n_pad)
    return ac[:max_size]


def tempogram(oenv: np.ndarray) -> np.ndarray:
    """win=344. Pad oenv con linear_ramp de 172 a cada lado. Frame hop=1.
       Para cada frame: x=opad[t:t+344]*hann344; tg[:,t]=AC_bounded(x)[:344].
       Normalizacion L-inf por COLUMNA (norm=np.inf, axis=-2)."""
    n = len(oenv)
    pad = WIN_LENGTH // 2  # 172
    # linear_ramp: rampa de 0 a oenv[0] en 'pad' pasos al inicio; de oenv[-1] a 0 al final
    head = np.linspace(0.0, oenv[0], pad + 2)[1:-1] if pad > 0 else np.array([])
    tail = np.linspace(oenv[-1], 0.0, pad + 2)[1:-1] if pad > 0 else np.array([])
    opad = np.concatenate([head, oenv, tail])
    w = hann_periodic(WIN_LENGTH)
    odf = np.empty((WIN_LENGTH, n), dtype=np.float64)
    for t in range(n):
        seg = opad[t:t + WIN_LENGTH] * w
        odf[:, t] = autocorrelate_bounded(seg, WIN_LENGTH)
    # norm L-inf por columna
    maxcols = np.max(np.abs(odf), axis=0, keepdims=True)
    maxcols[maxcols == 0] = 1.0   # evitar division por 0 (librosa normalize hace lo mismo)
    odf = odf / maxcols
    return odf


def tempo_estimate(oenv: np.ndarray) -> tuple:
    """tg_avg = mean_t(tg). bpms[0]=inf, bpms[L]=2583.984375/L.
       logprior[L]=-0.5*((log2(bpms[L])-log2(120)))^2; max_idx=argmax(bpms<320).
       logprior[:max_idx]=-inf. score=log1p(1e6*tg_avg)+logprior; tempo=bpms[argmax]."""
    tg = tempogram(oenv)
    tg_avg = np.mean(tg, axis=-1)                       # (344,)
    bpms = np.zeros(WIN_LENGTH, dtype=np.float64)
    bpms[0] = np.inf
    bpms[1:] = 60.0 * SR / (HOP * np.arange(1.0, WIN_LENGTH))
    logprior = -0.5 * ((np.log2(bpms) - np.log2(START_BPM)) / STD_BPM) ** 2
    max_idx = int(np.argmax(bpms < MAX_TEMPO))          # A3
    logprior[:max_idx] = -np.inf
    score = np.log1p(1e6 * tg_avg) + logprior           # A5
    best = int(np.argmax(score))
    return float(bpms[best]), tg_avg, bpms


# =============================================================================
#  Etapa 3 — DP beat tracker  (re-implementado, ya validado vs librosa)
# =============================================================================
def _normalize_onsets(oenv: np.ndarray) -> np.ndarray:
    norm = oenv.std(ddof=1, axis=-1, keepdims=True)     # ddof=1 (cuidado, numpy default es 0)
    return oenv / (norm + np.finfo(np.float64).tiny)


def _local_score(oenv: np.ndarray, fpb: float) -> np.ndarray:
    fpb = int(fpb)
    win = np.exp(-0.5 * (np.arange(-fpb, fpb + 1) * 32.0 / fpb) ** 2)
    return np.convolve(oenv, win, mode="same")


def _localmax(x: np.ndarray) -> np.ndarray:
    """A2: asimetrico. x[i]>x[i-1] estricto; x[i]>=x[i+1] no estricto. x[0] nunca."""
    m = np.zeros(len(x), dtype=bool)
    if len(x) >= 2:
        m[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
        m[-1] = x[-1] > x[-2]
    return m


def beat_track(oenv: np.ndarray, tempo: float, tightness: float = TIGHTNESS, trim: bool = True):
    frame_rate = float(SR) / HOP
    fpb = np.round(frame_rate * 60.0 / tempo).item()
    localscore = _local_score(_normalize_onsets(oenv), fpb)
    M = len(localscore)
    score_thresh = 0.01 * localscore.max()
    first_beat = True
    backlink = np.full(M, -1, dtype=np.int64)
    cumscore = np.zeros(M, dtype=np.float64)
    cumscore[0] = localscore[0]
    for i in range(1, M):
        best_score = -np.inf
        beat_location = -1
        loc = i - int(np.round(fpb / 2.0))
        hi = i - 2 * fpb - 1                       # A4: extremo inferior EXCLUSIVO
        while loc > hi:
            if loc < 0:
                break
            score = cumscore[loc] - tightness * (np.log(i - loc) - np.log(fpb)) ** 2
            if score > best_score:
                best_score = score
                beat_location = loc
            loc -= 1
        cumscore[i] = localscore[i] + best_score if beat_location >= 0 else localscore[i]
        if first_beat and localscore[i] < score_thresh:
            backlink[i] = -1
        else:
            backlink[i] = beat_location
            first_beat = False

    # tail
    lmax = _localmax(cumscore)
    masked_vals = cumscore.copy()
    masked_vals[~lmax] = np.nan                    # nan donde NO es max local -> queda para median
    # librosa: masked_array(data=cumscore, mask=~localmax) ; median ignora enmascarados
    # => mediana de los cumscore donde localmax es True
    med = np.nanmedian(cumscore[lmax])
    thr = 0.5 * med
    tail = M - 1
    while tail >= 0:
        if lmax[tail] and cumscore[tail] >= thr:
            break
        tail -= 1

    beats = np.zeros(M, dtype=bool)
    n = tail
    while n >= 0:
        beats[n] = True
        n = backlink[n]

    # trim
    w5 = hanning5()
    boe = localscore[beats]
    smooth = np.convolve(boe, w5)[2:len(localscore) + 2]
    thrT = 0.5 * np.sqrt(np.mean(smooth ** 2)) if trim else 0.0
    n = 0
    while n < M and localscore[n] <= thrT:
        beats[n] = False
        n += 1
    n = M - 1
    while n >= 0 and localscore[n] <= thrT:
        beats[n] = False
        n -= 1
    return localscore, cumscore, backlink, beats


# =============================================================================
#  Pipeline end-to-end
# =============================================================================
def run(y: np.ndarray):
    """y: float64 mono @22050. Devuelve dict con todos los intermedios."""
    S_power = stft_power(y)
    melfb = build_mel_filterbank()
    S_mel = melfb @ S_power
    S_dB = power_to_db(S_mel)
    oenv = onset_strength(S_dB)
    tempo, tg_avg, bpms = tempo_estimate(oenv)
    localscore, cumscore, backlink, beats = beat_track(oenv, tempo)
    beat_frames = np.flatnonzero(beats)
    beat_times = beat_frames * HOP / SR
    return dict(y=y, S_power=S_power, melfb=melfb, S_mel=S_mel, S_dB=S_dB,
                oenv=oenv, tempo=tempo, tg_avg=tg_avg, bpms=bpms,
                localscore=localscore, cumscore=cumscore, backlink=backlink,
                beats=beats, beat_frames=beat_frames, beat_times=beat_times)
