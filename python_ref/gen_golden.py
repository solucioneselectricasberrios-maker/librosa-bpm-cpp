"""
gen_golden.py — Genera el "golden master" del pipeline de librosa beat detection.

Vuelca a golden.npz TODOS los intermedios verificables:
  Etapa 1 (onset):  y, S_power, S_mel, S_dB, melfb, oenv, hann_stft, hann_tg
  Etapa 2 (tempo):  tg, tg_avg, bpms, tempo
  Etapa 3 (DP):     frames_per_beat, localscore, cumscore, backlink, beats
  Resultado:        beat_frames, beat_times

Los internos del beat tracker (localscore/cumscore/backlink) se re-implementan
aqui en numpy puro COPIANDO el fuente de librosa.beat._beat_tracker.__* (extraido
y verificado). De este modo el golden es completo y el C++ puede testear etapa
por etapa sin reconstruir nada.

Uso:
    python gen_golden.py
"""
from __future__ import annotations
import sys
import numpy as np
import librosa


# =============================================================================
#  Constantes del pipeline (defaults de librosa 0.11.0)
# =============================================================================
SR = 22050
HOP = 512
N_FFT = 2048
N_MELS = 128
FMAX = SR / 2.0        # 11025  (kwargs.setdefault("fmax", 0.5*sr) en onset_strength_multi)
AMIN = 1e-10
TOP_DB = 80.0
START_BPM = 120.0
STD_BPM = 1.0
AC_SIZE = 8.0
MAX_TEMPO = 320.0
TIGHTNESS = 100.0
WIN_LENGTH = int(np.floor(AC_SIZE * SR / HOP))   # 344


# =============================================================================
#  Re-implementacion numpy de los internos privados de librosa.beat
#  (fuente extraido de librosa/beat.py, funciones __beat_*)
# =============================================================================
def normalize_onsets(onsets: np.ndarray) -> np.ndarray:
    """onsets / (std(ddof=1) + tiny)"""
    norm = onsets.std(ddof=1, axis=-1, keepdims=True)
    return onsets / (norm + np.finfo(onsets.dtype).tiny)


def beat_local_score(onset_envelope: np.ndarray, frames_per_beat: float) -> np.ndarray:
    """Convolucion same-mode con gaussiana de ancho FPB. (copia de __beat_local_score, rama estatica)"""
    N = len(onset_envelope)
    fpb = int(frames_per_beat)
    window = np.exp(-0.5 * (np.arange(-fpb, fpb + 1) * 32.0 / fpb) ** 2)
    K = len(window)
    localscore = np.zeros(N, dtype=np.float64)
    for i in range(N):
        s = 0.0
        for k in range(max(0, i + K // 2 - N + 1), min(i + K // 2, K)):
            s += window[k] * onset_envelope[i + K // 2 - k]
        localscore[i] = s
    return localscore


def beat_track_dp(localscore: np.ndarray, frames_per_beat: float,
                  tightness: float = TIGHTNESS):
    """DP de Ellis 2007. Devuelve (backlink, cumscore). (copia de __beat_track_dp)"""
    M = len(localscore)
    score_thresh = 0.01 * localscore.max()
    first_beat = True
    backlink = np.full(M, -1, dtype=np.int64)
    cumscore = np.zeros(M, dtype=np.float64)
    cumscore[0] = localscore[0]

    fpb = frames_per_beat  # escalar
    for i in range(1, M):  # en librosa el enumerate empieza en 0 pero i=0 no entra al loop de loc
        best_score = -np.inf
        beat_location = -1
        lo = i - int(np.round(fpb / 2.0))
        hi = i - 2 * fpb - 1   # exclusivo
        loc = lo
        while loc > hi:
            if loc < 0:
                break
            score = cumscore[loc] - tightness * (np.log(i - loc) - np.log(fpb)) ** 2
            if score > best_score:
                best_score = score
                beat_location = loc
            loc -= 1

        if beat_location >= 0:
            cumscore[i] = localscore[i] + best_score
        else:
            cumscore[i] = localscore[i]

        if first_beat and localscore[i] < score_thresh:
            backlink[i] = -1
        else:
            backlink[i] = beat_location
            first_beat = False
    return backlink, cumscore


def localmax_asymmetric(x: np.ndarray) -> np.ndarray:
    """x[i] es max local si x[i] > x[i-1] (estricto) Y x[i] >= x[i+1] (no estricto).
    x[0] NUNCA es maximo. x[-1] lo es si x[-1] > x[-2]. (copia de util.localmax)"""
    n = len(x)
    m = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        m[i] = (x[i] > x[i - 1]) and (x[i] >= x[i + 1])
    if n >= 2:
        m[n - 1] = x[n - 1] > x[n - 2]
    return m


def last_beat(cumscore: np.ndarray) -> int:
    """Identifica la posicion del ultimo beat. (copia de __last_beat + __last_beat_selector)"""
    lmax = localmax_asymmetric(cumscore)
    mask = ~lmax                                    # mask=True donde NO es max local
    # mediana de los cumscore que SI son max locales (librosa masked median)
    peaks = cumscore[lmax]
    medians = np.median(peaks)
    threshold = 0.5 * medians
    # selector: ultimo n tal que (not mask[n]) and cumscore[n] >= threshold
    n = len(cumscore) - 1
    tail = n
    while n >= 0:
        if (not mask[n]) and cumscore[n] >= threshold:
            tail = n
            break
        n -= 1
    return int(tail)


def dp_backtrack(backlink: np.ndarray, tail: int, M: int) -> np.ndarray:
    beats = np.zeros(M, dtype=bool)
    n = tail
    while n >= 0:
        beats[n] = True
        n = backlink[n]
    return beats


def trim_beats(localscore: np.ndarray, beats: np.ndarray, trim: bool = True) -> np.ndarray:
    """Recorta beats debiles. (copia de __trim_beats)"""
    beats_trimmed = beats.copy()
    w = np.hanning(5)   # [0, 0.5, 1.0, 0.5, 0]
    smooth_boe = np.convolve(localscore[beats], w)[len(w) // 2:len(localscore) + len(w) // 2]
    threshold = 0.5 * (np.mean(smooth_boe ** 2) ** 0.5) if trim else 0.0

    n = 0
    while n < len(localscore) and localscore[n] <= threshold:
        beats_trimmed[n] = False
        n += 1
    n = len(localscore) - 1
    while n >= 0 and localscore[n] <= threshold:
        beats_trimmed[n] = False
        n -= 1
    return beats_trimmed


def full_beat_tracker(oenv: np.ndarray, tempo: float, frame_rate: float,
                      tightness: float = TIGHTNESS, trim: bool = True) -> dict:
    frames_per_beat = np.round(frame_rate * 60.0 / tempo).item()
    localscore = beat_local_score(normalize_onsets(oenv), frames_per_beat)
    backlink, cumscore = beat_track_dp(localscore, frames_per_beat, tightness)
    tail = last_beat(cumscore)
    beats = dp_backtrack(backlink, tail, len(oenv))
    beats = trim_beats(localscore, beats, trim)
    return {
        "frames_per_beat": frames_per_beat,
        "localscore": localscore,
        "cumscore": cumscore,
        "backlink": backlink,
        "tail": tail,
        "beats_bool": beats,
    }


# =============================================================================
#  Pipeline completo
# =============================================================================
def main(mp3_path: str, out_npz: str, out_json: str):
    print(f"[golden] loading {mp3_path}")
    y, sr = librosa.load(mp3_path, sr=SR, mono=True)
    y = y.astype(np.float64)   # promover a float64 para todo el pipeline
    print(f"[golden] N samples = {len(y)}  (M onset = {1 + len(y)//HOP})")

    # ---- Etapa 1: onset ----
    from scipy.signal import get_window
    hann_stft = get_window("hann", N_FFT, fftbins=True).astype(np.float64)
    hann_tg = get_window("hann", WIN_LENGTH, fftbins=True).astype(np.float64)

    melfb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS,
                                fmin=0.0, fmax=FMAX, htk=False, norm="slaney").astype(np.float64)

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann",
                            center=True, pad_mode="constant"))
    S_power = (S ** 2).astype(np.float64)
    S_mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                           n_mels=N_MELS, fmax=FMAX, htk=False,
                                           norm="slaney", power=None)
    if S_mel.shape[0] != N_MELS or S_mel.shape[1] != S_power.shape[1]:
        # melspectrogram con S de entrada ya no re-aplica el filtro; lo hacemos explicito
        S_mel = melfb @ S_power
    S_dB = librosa.power_to_db(S_mel, amin=AMIN, top_db=TOP_DB).astype(np.float64)

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP,
                                        aggregate=np.median).astype(np.float64)
    print(f"[golden] oenv shape={oenv.shape} mean={oenv.mean():.6f} std={oenv.std():.6f}")

    # ---- Etapa 2: tempo ----
    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=HOP,
                                   win_length=WIN_LENGTH).astype(np.float64)
    tg_avg = np.mean(tg, axis=-1).astype(np.float64)
    bpms = librosa.tempo_frequencies(WIN_LENGTH, hop_length=HOP, sr=sr).astype(np.float64)
    tempo = float(librosa.feature.tempo(onset_envelope=oenv, sr=sr,
                                        hop_length=HOP, start_bpm=START_BPM,
                                        std_bpm=STD_BPM, max_tempo=MAX_TEMPO,
                                        aggregate=np.mean)[0])
    print(f"[golden] tempo = {tempo:.6f} BPM")

    # ---- Etapa 3: DP (re-implementado, validado vs librosa) ----
    frame_rate = float(sr) / HOP
    bt = full_beat_tracker(oenv, tempo, frame_rate, TIGHTNESS, trim=True)
    beat_frames = np.flatnonzero(bt["beats_bool"])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    print(f"[golden] beats = {len(beat_frames)}")
    print(f"[golden] first beat = {beat_times[0]:.5f}s   last = {beat_times[-1]:.5f}s")

    # ---- Volcado ----
    np.savez(out_npz,
             y=y, S_power=S_power, S_mel=S_mel, S_dB=S_dB,
             melfb=melfb, hann_stft=hann_stft, hann_tg=hann_tg,
             oenv=oenv,
             tg=tg, tg_avg=tg_avg, bpms=bpms, tempo=np.float64(tempo),
             win_length=np.int64(WIN_LENGTH),
             frames_per_beat=np.float64(bt["frames_per_beat"]),
             localscore=bt["localscore"], cumscore=bt["cumscore"],
             backlink=bt["backlink"], tail=np.int64(bt["tail"]),
             beats_bool=bt["beats_bool"],
             beat_frames=beat_frames, beat_times=beat_times)
    print(f"[golden] saved {out_npz}")

    import json
    json.dump({
        "filepath": "song_f32.wav", "sr_analysis": SR, "hop_length": HOP, "n_fft": N_FFT,
        "n_samples": len(y), "n_frames_onset": len(oenv),
        "global_tempo": tempo, "n_beats": int(len(beat_frames)),
        "beat_times": beat_times.tolist(),
    }, open(out_json, "w"), indent=2)
    print(f"[golden] saved {out_json}")


if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AUDIO = os.path.join(ROOT, "tests", "golden", "song_f32.wav")
    OUT_NPZ = os.path.join(ROOT, "tests", "golden", "golden.npz")
    OUT_JSON = os.path.join(ROOT, "tests", "golden", "golden_beats.json")
    main(AUDIO, OUT_NPZ, OUT_JSON)
