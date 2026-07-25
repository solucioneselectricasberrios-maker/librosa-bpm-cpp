# 📐 FÓRMULAS VERIFICADAS — Referencia rápida

> Todas las fórmulas de abajo están **verificadas empíricamente** contra el código fuente
> de librosa 0.11.0 (ejecutándolo, no leyendo la doc). Cada una tiene su valor numérico
> real para que puedas contrastar tu implementación.

---

## Parámetros del pipeline (defaults de librosa, usados por NOISE)

| Parámetro | Valor | Notas |
|-----------|-------|-------|
| `sr` | 22050 | sample rate de análisis |
| `hop_length` | 512 | |
| `n_fft` | 2048 | |
| `n_mels` | 128 | |
| `fmin` / `fmax` | 0 / 11025 | `fmax = sr/2` (onset_strength_multi) |
| `lag` | 1 | spectral flux lag |
| `start_bpm` | 120.0 | prior |
| `std_bpm` | 1.0 | prior (octavas) |
| `ac_size` | 8.0 s | ventana tempogram |
| `max_tempo` | 320.0 | |
| `tightness` | 100.0 | DP |
| `trim` | true | |
| `aggregate` (onset) | `np.median` | forzado por `beat_track` |
| `top_db` | 80.0 | en `power_to_db` |
| `amin` | 1e-10 | en `power_to_db` |

**Derivados:**
- `win_length = floor(8·sr/hop) = 344`
- `win_half = 172`
- `frame_rate = sr/hop = 43.06640625`
- `pad_onset = lag + n_fft/(2·hop) = 1 + 2 = 3`
- `M (frames) = 1 + N//hop` (ej: N=661,500 → M = 1,292)

---

## 1. Ventanas

### Hann periódica (`fftbins=True`) — denominador **N**
```
w[n] = 0.5 - 0.5·cos(2π·n/N)     para n = 0..N-1
```
- STFT: `N = 2048` → `w[0]=0, w[1]=2.353e-06, w[1024]=1.0, w[2047]=2.353e-06`
- Tempogram: `N = 344`

### `np.hanning(5)` (para el trim de beats)
```
[0.0, 0.5, 1.0, 0.5, 0.0]
```

---

## 2. Escala Mel (Slaney)
```
f <  1000 Hz:  mel = f / 66.66666666666667
f >= 1000 Hz:  mel = 15.0 + ln(f/1000) / 0.06875177742094912
```
- `0.06875177742094912 = ln(6.4)/27`
- `hz_to_mel(1000) = 15.0`
- `hz_to_mel(11025) = 49.91059448`
- Inversa: `mel < 15 → mel·66.6667`; `mel >= 15 → 1000·exp(0.06875178·(mel-15))`

### Frecuencias centrales del melbank
130 puntos uniformes en Mel entre `hz_to_mel(0)` y `hz_to_mel(11025)`, convertidos a Hz.
- `mel_f[0] = 0`, `mel_f[1] = 25.794`, `mel_f[129] = 11025`

---

## 3. Melbank triangular (Slaney)
Para cada filtro `i = 0..127`, para cada bin FFT `k = 0..1024` con `fftfreqs[k] = k·sr/n_fft`:
```
fd0 = mel_f[i+1] - mel_f[i]
fd1 = mel_f[i+2] - mel_f[i+1]
lower = (fftfreqs[k] - mel_f[i])     / fd0      ← asciende 0→1   (¡NO invertir!)
upper = (mel_f[i+2] - fftfreqs[k])   / fd1      ← desciende 1→0
H[i][k] = max(0, min(lower, upper)) · (2.0 / (mel_f[i+2] - mel_f[i]))
```
- Suma total del melbank (sanity check) ≈ **11.8867**
- La fórmula del plan original (A/B invertidas) daba **suma 0** → completamente rota.

---

## 4. STFT → espectro de potencia
```
y_padded = [0]·1024 + y + [0]·1024            (center=True, pad_mode='constant')
M = 1 + len(y)//hop
Para cada frame m = 0..M-1:
  frame[n] = y_padded[m·hop + n] · hann[n]     n = 0..2047
  X = RFFT(frame, n=2048)                       → 1025 bins
  S_power[k,m] = X.real[k]² + X.imag[k]²       k = 0..1024
```

---

## 5. power_to_db (con `top_db=80`)
```
log_spec = 10·log10(max(amin, S))              amin=1e-10, ref=1.0
log_spec -= 10·log10(max(amin, 1.0))
log_spec = max(log_spec, log_spec.max() - 80)  ← clamp top_db (¡no omitir!)
```

---

## 6. Onset strength envelope (`aggregate=np.median`)
```
flux[i,m] = max(0, S_dB[i,m] - S_dB[i,m-1])    m = 1..M-1
oenv_raw[m-1] = median_128(flux[:,m])          ← mediana de 128 (par): media de idx 63 y 64
oenv = [0,0,0] + oenv_raw                      ← pad_onset = 3
oenv = oenv[0:M]                               ← trim a M
```
**Valores verificados (canción de prueba):**
- `oenv.mean() = 0.70235157`, `oenv.std() = 1.41692615` (ddof=0)
- `oenv[0:10] = 0` (silencio al inicio)

---

## 7. Tempogram (autocorrelación enventanada)
```
opad = linear_ramp_pad(oenv, 172)              ← rampa 0→oenv[0] y oenv[-1]→0 en 172 pasos
w = hann_periodic(344)
Para cada t = 0..M-1:
  x[τ] = opad[t+τ] · w[τ]                      τ = 0..343
  tg[:,t] = AC_bounded(x)[:344]
  donde AC_bounded: n_pad = next_fast_len(2·344-1) = 720
                     ac = irfft(|rfft(x, n=720)|², n=720)[:344]
                     (|·|² como parte real, parte imag 0)
Normalización L∞ por COLUMNA: tg[:,t] /= max|tg[:,t]|
```

### `next_fast_len(687)` = **720** (menor ≥687 con factores 2,3,5,7)

---

## 8. Estimación de tempo
```
tg_avg[τ] = mean_t(tg[τ,t])                    τ = 0..343
bpms[0] = +inf
bpms[L] = 60·sr / (hop·L) = 2583.984375 / L    L = 1..343
logprior[L] = -0.5·((log2(bpms[L]) - log2(120)))²
max_idx = argmax(bpms < 320)                   ← primer L con bpm < 320
logprior[0..max_idx-1] = -inf
score[L] = log1p(1e6 · tg_avg[L]) + logprior[L]
tempo = bpms[argmax(score)]
```
- Para L donde `bpms > 320`: quedan a `-inf` (el prior los mata).
- `argmax` devuelve el **primer** máximo en caso de empate.

**Valor verificado:** `tempo = 117.453835 BPM` (audio de prueba)

---

## 9. DP beat tracker (Ellis 2007)

### Normalización de onsets
```
σ = std(oenv, ddof=1)                          ← ddof=1 (¡numpy default es 0!)
on[t] = oenv[t] / (σ + tiny)
```
- `tiny` = `finfo(dtype).tiny`. Operando en float64 → `2.225e-308`.

### Local score (gaussian blur)
```
FPB = round(frame_rate·60 / tempo)             = round(2583.984375/tempo)  → 25
sigma efectivo: kernel usa k·32/FPB
win[k] = exp(-0.5·(k·32/FPB)²)                 k = -FPB..FPB  (longitud 2·FPB+1 = 51)
localscore = convolve(on, win, mode='same')
```

### DP (núcleo)
```
score_thresh = 0.01 · max(localscore)
cumscore[0] = localscore[0]; backlink[0] = -1; first_beat = true
Para i = 1..M-1:
  mejor = -inf; loc_mejor = -1
  para loc = i - round(FPB/2) hasta i - 2·FPB - 1 (step -1):   ← extremo inferior EXCLUSIVO
    si loc < 0: break
    score = cumscore[loc] - 100·(ln(i-loc) - ln(FPB))²
    si score > mejor: mejor = score; loc_mejor = loc
  cumscore[i] = localscore[i] + mejor   (si loc_mejor>=0, si no solo localscore[i])
  si first_beat y localscore[i] < score_thresh: backlink[i] = -1
  si no: backlink[i] = loc_mejor; first_beat = false
```

### Tail + backtracking
```
lmax = localmax_asimétrico(cumscore)           ← x[0] nunca es máximo
med = median(cumscore[lmax])                   ← mediana solo de los picos
thr = 0.5 · med
tail = último n tal que lmax[n] y cumscore[n] >= thr
backtrack: beats[tail]=1; seguir backlink hasta -1
```
**Valor verificado:** `FPB = 25`, `tail = 12320`, backtracking pre-trim = 496 candidatos.

### Trim
```
w5 = [0, 0.5, 1.0, 0.5, 0]                     ← np.hanning(5)
boe = localscore[beats]                        ← solo en posiciones de beat
smooth = convolve(boe, w5)[2 : 2+M]            ← recorte
thrT = 0.5 · sqrt(mean(smooth²))
eliminar beats iniciales mientras localscore[n] <= thrT
eliminar beats finales   mientras localscore[n] <= thrT
```
**Resultado:** 59 beats detectados (click track 120 BPM).

---

## 10. Conversión frames → segundos
```
beat_times[i] = beat_frame[i] · hop / sr = beat_frame[i] · 0.0232199546...
```
**Valores verificados:**
- Primer beat: frame 24 → `0.55728s`
- Último beat: frame 12104 → `281.05433s`

---

## 11. localmax asimétrico (detalle crucial)
```
x[i] es máximo local si:
   x[i] >  x[i-1]      (estricto)
   x[i] >= x[i+1]      (no estricto)
x[0]   NUNCA es máximo
x[-1]  lo es si x[-1] > x[-2]
```
Determina el `tail` del backtracking. Si lo implementás simétrico estándar, elegís un tail
distinto y todos los beats cambian.

---

## 12. Tolerancias calibradas (para los tests C++)

| Etapa | Métrica | Tolerancia |
|-------|---------|------------|
| Ventanas | max abs diff | < 1e-12 (ruido ULP) |
| Melbank | max abs diff | < 1e-8 (el enorm amplifica ULP) |
| STFT potencia | MAE bloque 10×10 | < 1e-6 |
| Mel spec | MAE bloque 10×10 | < 1e-5 |
| dB | max abs diff | < 0.01 dB |
| Onset env | correlación | > 0.9999 |
| tg_avg | correlación | > 0.999 |
| Tempo | abs diff | < 0.01 BPM |
| localscore | correlación | > 0.999 |
| cumscore | correlación | > 0.999 |
| backlink | % idéntico | > 99% |
| beats | array bool | identico |
| E2E | similarity @0.04s | ≥ 93% (conseguido: 100%) |
