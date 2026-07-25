# 🔍 Análisis crítico de la Sección 8 + Plan de Tests

**Generado:** 2026-06-30 · **Verificado contra:** librosa 0.11.0, numpy 2.4.6, scipy 1.18.0
**Audio de referencia:** 661,500 muestras @ 22,050 Hz (30.0s)

Cada afirmación se verificó ejecutando el código real de librosa, no leyendo la doc.
Leyenda: ✅ correcto · ⚠️ matiz peligroso · ❌ ERRÓNEO (rompería la similitud).

---

## ❌ 8.2 — La ventana Hann periódica (ERROR GRAVE)

El doc dice:
> `w[n] = 0.5 - 0.5 * cos(2π * n / (N + 1))`  ← denominador **N+1**

**Verificación empírica contra `scipy.signal.get_window('hann', 2048, fftbins=True)`:**

```
w[0]    = 0.0
w[1]    = 2.353e-06
w[1024] = 1.0
w[2047] = 2.353e-06

match con 0.5 - 0.5*cos(2π·n/N)      → True   ✅
match con 0.5 - 0.5*cos(2π·n/(N+1))  → False  ❌
```

**Fórmula correcta:** `w[n] = 0.5 - 0.5·cos(2π·n / N)` con `N = win_length`.
- STFT: N = 2048
- Tempogram: N = 344
- Trim (5 puntos): es `np.hanning(5)` → ver §8.6.

> ⚠️ Confusión de origen: scipy tiene DOS fórmulas Hann. La *simétrica* (`fftbins=False`) usa `N-1`;
> la *periódica* (`fftbins=True`, la que usa librosa) usa **`N`**. El doc mezcló ambos.
> Esto desplaza los picos del onset ~1 frame y, acumulado en el DP, puede tirar beats completos.

---

## ❌ 8.4 — La cuenta de frames M (ERROR)

El doc da **dos fórmulas distintas y ambas incorrectas**:
- §2.2: `M = ceil((N + 1024) / 512)`  → da 12,323
- §8.4: `M = (N + 2047)//512 + 1`     → da 12,325

**Verificacion empirica** (con N = 661,500 y arrays sinteticos):

| N        | M real (stft) | M real (oenv) | `1+N//512` | `ceil((N+1024)/512)` |
|----------|---------------|---------------|------------|----------------------|
| 661500  | **1292**     | **1292**     | **1292**  | 1293 ❌              |
| 10000    | **20**        | **20**        | **20**     | 22 ❌                 |
| 2048     | **5**         | **5**         | **5**      | 6 ❌                  |

**Fórmula correcta:** con `center=True, pad_mode='constant'`, librosa rellena `n_fft//2 = 1024`
ceros a **cada lado** (no solo a uno) y luego enmarca, de modo que:

```
M = 1 + ( (N + 2·1024) - 2048 ) // 512  =  1 + N // 512
```

> El valor objetivo del doc (12,321) solo "coincide" porque con la fórmula incorrecta
> los redondeos casan para ese N concreto. Cualquier otro audio daría longitudes distintas y
> el onset envelope se desalinearía → offsets de frames en TODOS los beats.

---

## ❌ 8.6 — Ventana Hann de 5 puntos para el trim (ERROR)

El doc dice:
> `w_smooth = [0.0, 0.345, 0.655, 0.345, 0.0]`

**Verificación:** `np.hanning(5)` devuelve **`[0.0, 0.5, 1.0, 0.5, 0.0]`**, NO `[0,0.345,0.655,0.345,0]`.

Los valores `0.345/0.655` corresponden a la fórmula Hann periódica **mal aplicada** con N=5
(`0.5-0.5·cos(2π·k/5)` → 0.345, 0.655…), pero `np.hanning` usa la simétrica `cos(2π·k/(N-1))`
→ 0.5, 1.0, 0.5. Esto cambia el `smooth_boe` y por tanto el umbral de trim.

> Impacto: medio. Afecta solo al recorte de beats inicial/final, pero puede costar 1–3 beats
> en los bordes y bajar la similitud por debajo de 93%.

---

## ⚠️ 8.8 — epsilon en normalización (MATIZ PELIGROSO)

El doc dice usar `tiny(float64) = 2.225e-308`. **Sutilmente correcto pero engañoso.**

Verificación: `__normalize_onsets` hace:
```python
norm = onsets.std(ddof=1, ...)      # ← ddof=1, ver §3.2 abajo
return onsets / (norm + util.tiny(onsets))
```
y `util.tiny(x)` depende de **`x.dtype`**, no es siempre float64. En la práctica el onset
envelope de librosa es **float32** → `tiny = 1.175e-38`, no `2.225e-308`.

> Si fuerzas float64 porque "es más preciso", cambias el epsilon y, en señales casi silentes,
> el cociente difiere. **Regla segura:** computa TODO en float64 (como dice §8.1) pero usa
> `tiny` del MISMO dtype con el que divides. Lo más limpio: operar en float64 de extremo a extremo
> (onset_env incluido) → entonces sí `2.225e-308` es correcto. **No mezcles.**

---

## ⚠️ §3.2 / 8.x — `std` usa `ddof=1` (EL DOC LO DICE BIEN, PERO FÁCIL DE EQUIVOCAR)

Confirmado del fuente: `onsets.std(ddof=1, axis=-1, keepdims=True)`.

NumPy por defecto es `ddof=0`. Si en C++ usas la desviación típica poblacional (divides por N
en vez de por N-1) → la normalización del onset se escala por factor `sqrt(N/(N-1))`,
~despreciable para N=M_frames, **pero no para el localscore de ventanas cortas**. Usar `ddof=1`.

---

## ✅ Aciertos importantes del doc (para que no los "corrijas" por error)

| Ítem | Doc | Verificado |
|------|-----|------------|
| `win_length = floor(8·sr/hop) = 344` | 344 | ✅ exacto |
| `next_fast_len(687, real=True)` | "720 o 700" | ✅ exactamente **720** |
| `aggregate=np.median` en onset | sí | ✅ `beat_track` lo fuerza explícitamente |
| Mel Slaney: `f/66.6667` y `15+ln(f/1000)/0.06875178` | sí | ✅ `ln(6.4)/27 = 0.06875177…` |
| `pad_mode='constant'`, `center=True` | sí | ✅ defaults de `stft` |
| Mediana de 128 = media de idx 63 y 64 | sí | ✅ |
| `bpm[L] = 60·sr/(hop·L) = 2583.984375/L` | sí | ✅ |
| Penalización DP `tightness·(ln(Δ)-ln(FPB))²` | sí | ✅ del fuente `__beat_track_dp` |
| `log1p(1e6·tg)` en el score de tempo | sí | ✅ |

---

## 🔴 DETALLES QUE EL DOC NO MENCIONA Y SÍ ROMPEN LA SIMILITUD

### A. `localmax` es ASIMÉTRICO y `x[0]` nunca es máximo
```python
x[i] es máximo local si:  x[i] > x[i-1]  (estricto)
                         y x[i] >= x[i+1] (no estricto)
# x[0] NUNCA se considera máximo (no cumple la 1ª condición)
# x[-1] lo es si x[-1] > x[-2]
```
Esto se usa en `__last_beat` para enmascarar el cumscore y hallar el "tail". Si implementas
un máximo local simétrico estándar (`x[i] >= x[i-1] and x[i] >= x[i+1]`), eliges un tail
distinto → toda la cadena de backtracking cambia.

### B. `max_idx` del prior se calcula con `argmax(bpms < max_tempo)`, no "bpms > 320"
```python
max_idx = int(np.argmax(bpms < max_tempo))   # max_tempo=320
logprior[:max_idx] = -np.inf
```
`bpms[0] = +inf` (0-lag). `argmax(bpms<320)` = primer índice donde bpm<320.
Como `bpms` es decreciente en L, **todo L con bpm≥320 queda a -inf**.
No es "recortar L donde bpm>320" literal; hay que replicar el `argmax` exacto con `bpms[0]=inf`.

### C. `frames_per_beat` es un **array**, el DP indexa `frames_per_beat[tv·i]`
```python
frames_per_beat = np.round(frame_rate*60/bpm)   # escalar → array de 1 elemento
tv = int(len(fpb) > 1)                           # tv=0 en modo tempo estático
range(i - round(fpb[0]/2), i - 2*fpb[0] - 1, -1)  # límites: nota el -1 (exclusivo)
```
El bucle de predecesores es `range(i - round(FPB/2), i - 2·FPB - 1, step=-1)`.
El `-1` final hace que el extremo inferior **sea exclusivo** (no se evalúa `i-2·FPB`).
Fácil equivocarse aquí al portar a C++.

### D. `score_thresh` y el "first beat"
```python
score_thresh = 0.01 * localscore.max()
# first_beat=True inicialmente. Solo se pone False cuando score_i >= thresh
# Mientras first_beat: backlink=-1 (no acumula), cumscore=score_i
```
El doc lo describe pero mezcla el orden. Orden real en cada frame i:
1. buscar mejor predecesor loc
2. `cumscore[i] = score_i + best_score` (si hubo loc) ó `score_i`
3. **si** `first_beat and score_i < thresh`: `backlink[i] = -1`
   **sino**: `backlink[i] = beat_location; first_beat = False`

### E. Trim: el umbral usa `localscore[beats]`, NO `localscore[t]` directo
```python
smooth_boe = np.convolve(localscore[beats], np.hanning(5))[2:len+2]
threshold = 0.5 * sqrt(mean(smooth_boe**2))
# luego poda extremos mientras localscore[n] <= threshold
```
`localscore[beats]` = solo los valores de localscore **en posiciones de beat**.
Convolucionar eso con hanning(5) y luego RMS·0.5. Fácil equivocar el slicing
`[len(w)//2 : len(localscore)+len(w)//2]`.

### F. `power_to_db` usa `10·log10(max(amin, S))`, NO `10·log10(S)` con clamp separado
Y `amin` por defecto es **1e-10** ✅ (el doc acertó), pero el `top_db` default (80) **NO se aplica**
aquí porque `onset_strength_multi` llama a `core.power_to_db(S)` **sin** `top_db` explícito →
usa default 80. **Esto el doc NO lo dice.** Verificar: ¿se aplica top_db=80?
```python
S = core.power_to_db(S)   # top_db=80 default → clamp del rango dinámico a 80 dB
```
→ **Sí, top_db=80 está activo.** Hay que replicarlo o el spectral flux cambia.

---

# 🧪 PLAN DE TESTS (estrategia verificable)

Filosofía: **golden-master testing** contra numpy/librosa real. Cada etapa se valida
**antes** de avanzar. Tolerancias calibradas empíricamente (no inventadas).

## Etapa 0 — Generación del golden master (Python)
Script `gen_golden.py` que vuelca a `golden.npz`:
- `y` (audio float32 @22050, mono) — **incluyendo** los samples exactos tras `librosa.load`.
- `S_mel` (128×M_frames), `S_dB` (128×M_frames), `oenv` (M_frames,), `tempo` (escalar),
  `beat_frames` (variable), `localscore`, `cumscore`, `backlink`, `frames_per_beat`.
- además `melfb` (128×1025) y `hann_stft` (2048,), `hann_tg` (344,) para validar ventanas.

## Etapa 1 — Ventanas y constantes
- **Test Hann:** comparar `hann_stft` y `hann_tg` contra `0.5-0.5·cos(2π·n/N)`. Tolerancia **0** (bit-exacto).
- **Test np.hanning(5):** `[0,0.5,1,0.5,0]` bit-exacto.

## Etapa 2 — Carga de audio + frame count
- **Test M:** con `golden.y`, `M == 1 + len(y)//512 == M_frames`. Falla si ≠.
- **Test audio:** los primeros/últimos 10 samples de C++ vs Python, error relativo < 1e-6.
  (El MP3 decoding con `dr_mp3` puede diferir de ffmpeg → **este es el mayor riesgo**; ver nota abajo.)

## Etapa 3 — STFT + Mel + dB
- Comparar `S_mel[0:10,0:10]` C++ vs Python: MAE relativo < **1e-5**.
- Comparar `S_dB[0:10,0:10]`: error abs < **0.01 dB** (incluye `top_db=80`).
- **Crítico:** la FFT debe usar el **mismo** modelo numérico que numpy (RFFT de 2048 → 1025 bins).
  KissFFT cumple; validar con un impulso unitario: `|RFFT([1,0,...,0])| == [1,1,...,1]`.

## Etapa 4 — Onset envelope
- Comparar `oenv` completo (M_frames valores): correlación > **0.9999** y MAE relativo < **1e-4**.
- Esta es la métrica más discriminante. Si pasa, etapas 1–4 están bien.

## Etapa 5 — Tempo
- `tempo` dentro de **+-0.01 BPM** del golden.
- Comparar `tg_avg` (344 valores) con correlación > 0.999.
- Validar `max_idx` del prior manualmente con `bpms`.

## Etapa 6 — Beat tracking
- **Test de trazado:** reproducir `localscore`, `cumscore`, `backlink` contra golden con:
  correlación > 0.999 en localscore, y `backlink` idéntico en ≥99% de frames.
- **Test tail:** mismo `tail` que Python (depende de `localmax` asimétrico).
- **Test trim:** mismo `beats` boolean tras el trim.

## Etapa 7 — End-to-end
- `similarity_score(cpp_beats, py_beats, tol=0.04s) ≥ 93%`.
- Bonus: misma métrica a `tol=0.02s` (≈1 frame) como regresión estricta interna.

## Nota de riesgo: el decodificador MP3
`librosa.load` (vía ffmpeg) y `dr_mp3` **no** dan samples bit-iguales. Diferencias ~1 LSB en
float32 se acumulan en el STFT. Estrategia:
1. Generar también un **WAV** de la misma canción (`librosa.load` → `soundfile.write` WAV 32-bit)
   y probar el pipeline C++ contra **ese WAV** primero (dr_wav, bit-exacto). Esto aísla la lógica
   del algoritmo del problema de decodificación.
2. Una vez ≥93% con WAV, re-introducir MP3 (dr_mp3) y medir la degradación.
3. Si con MP3 no se llega a 93%, considerar `librosa.load` del MP3 → volcar PCM crudo →
   consumir ese PCM en C++ (evita el decodificador). Es lo más fiable.

---

# 📌 RESUMEN DE ACCIÓN (qué cambiar en el plan original)

1. **Corregir §2.2, §5 y §8.2:** Hann periódica = `0.5-0.5·cos(2π·n/N)` (N, no N+1 ni N-1).
2. **Corregir §2.2 y §8.4:** `M = 1 + N//hop` (con N muestras @22050 tras load).
3. **Corregir §8.6:** ventana de trim = `np.hanning(5) = [0,0.5,1,0.5,0]`.
4. **Añadir §8.x:** `top_db=80` activo en `power_to_db` (NO documentado en el original).
5. **Añadir §8.x:** `localmax` asimétrico, `x[0]` nunca máximo.
6. **Añadir §8.x:** `max_idx = argmax(bpms < 320)` con `bpms[0]=+inf`.
7. **Añadir §8.x:** el rango del DP es `range(i-round(FPB/2), i-2·FPB-1, -1)` (extremo inferior **exclusivo**).
8. **Mantener** `ddof=1`, `np.median`, `log1p(1e6·tg)`, `next_fast_len=720`, Mel Slaney, `pad_mode='constant'`.
9. **Tests:** golden-master WAV-first, tolerancias calibradas, metric `tol=0.04s`.
