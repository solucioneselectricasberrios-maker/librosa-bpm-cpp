# 🏗️ PLAN DE IMPLEMENTACIÓN EN PROFUNDIDAD — Port de librosa beat detection a C++

**Fecha:** 2026-06-30 · **Target:** ≥93% similitud en beats vs `reference_python.json`
**Verificado contra:** librosa 0.11.0 · numpy 2.4.6 · scipy 1.18.0 (código fuente, no docs)
**Workspace:** `C:\Opencode\New folder\`

> ⚠️ Este plan **corrige 4 fórmulas erróneas** del `LIBROSA_BPM_PORT_CPP.md` original y
> **añade 5 detalles omitidos** que romperían la similitud. Todas las fórmulas de abajo
> están verificadas empíricamente contra el código real de librosa. Ver
> `SECCION8_ANALISIS_CRITICO.md` para el detalle de cada corrección.

---

## 0. CORRECCIONES CRÍTICAS (a aplicar ANTES de programar nada)

| # | Errónea en el doc | Correcta (verificada) |
|---|------------------|------------------------|
| **C1** | Hann: `0.5-0.5·cos(2π·n/(N+1))` | `0.5-0.5·cos(2π·n/N)` (denominador **N**) |
| **C2** | `M = ceil((N+1024)/512)` / `(N+2047)//512+1` | `M = 1 + N//hop` (con N = muestras tras `load`) |
| **C3** | Trim window `[0,0.345,0.655,0.345,0]` | `np.hanning(5) = [0,0.5,1,0.5,0]` |
| **C4** | Melbank: `A=(f₁-f)/(f₁-f₀), B=(f-f₁)/(f₂-f₁)` | `lower=(f-f₀)/(f₁-f₀), upper=(f₂-f)/(f₂-f₁)` con `min(lower,upper)` — las del doc dan suma 0 |

**Añadidos omitidos por el doc:**
- **A1.** `power_to_db` con `top_db=80` ACTIVO → clamp `max(log_spec) − 80`.
- **A2.** `localmax` **asimétrico**: `x[i]>x[i-1]` (estricto) ∧ `x[i]>=x[i+1]` (no estricto); `x[0]` **nunca** es máximo.
- **A3.** `max_idx = argmax(bpms < 320)` con `bpms[0]=+inf` (no "bpm>320" literal).
- **A4.** Bucle de predecesores del DP: `range(i-round(FPB/2), i-2·FPB-1, -1)` → extremo inferior **exclusivo**.
- **A5.** `std(ddof=1)`, `tiny` del mismo dtype que el operando, `log1p(1e6·tg)`.

---

## 1. STACK TECNOLÓGICO (justificado)

| Componente | Elección | Razón |
|-----------|----------|-------|
| FFT | **KissFFT** (float64, `kiss_fftr`) | single-header, soporta cualquier tamaño, fácil de igualar a `scipy.fft.rfft`. **Alternativa:** FFTW3f si la velocidad es crítica. |
| Audio IO (WAV) | **dr_wav.h** | bit-exacto vs `soundfile` para los tests golden-master. |
| Audio IO (MP3) | **dr_mp3.h** | single-header. ⚠️ No es bit-exacto vs ffmpeg → ver §10 (riesgo). |
| Álgebra | **std::vector<double>** + struct `Matrix` minimal | Evitar arrastrar Eigen solo para multiplicación matriz-vector. |
| Build | **CMake** +FetchContent | Descarga KissFFT y dr_* automáticamente. |
| Tests | **GoogleTest** (o catch2) + golden `.npz`/`.json` | |
| dtype interno | **float64 de extremo a extremo** | Coincide con numpy por defecto; evita el problema del `tiny`. Solo el audio PCM se lee float32 y se promueve. |

**Sin dependencias externas binarias:** todo entra como header único o submodulo de git. Cero installers.

---

## 2. ARQUITECTURA DEL PROYECTO

```
C:\Opencode\New folder\
├── CMakeLists.txt
├── third_party/
│   ├── kissfft/                  (submódulo)
│   ├── dr_mp3.h
│   ├── dr_wav.h
│   └── nlohmann/json.hpp         (para I/O de salida)
├── include/librosa_bpm/
│   ├── constants.h               # π, SR, HOP, N_FFT, N_MELS, defaults
│   ├── math_util.h               # hz_to_mel, mel_to_hz, hann, hanning5, median
│   ├── matrix.h                  # Matrix<T> row-major + gemm mel·spec
│   ├── audio_loader.h            # dr_wav/dr_mp3 → mono float64 @22050
│   ├── stft.h                    # RFFT 2048, |·|², centering
│   ├── mel_filterbank.h          # Slaney 128×1025
│   ├── power_to_db.h             # 10·log10(max(amin,S)) − ... ; top_db=80
│   ├── onset_detector.h          # flux + median + pad(3) → oenv
│   ├── tempogram.h               # win=344, AC por FFT, norm L∞ por columna
│   ├── tempo_estimator.h         # tg_avg + prior + log1p → argmax
│   ├── beat_tracker.h            # local_score + DP + backlink + trim
│   ├── beat_utils.h              # frames_to_time, local_bpms
│   └── bpm_detector.h            # API pública (orquesta todo)
├── src/  (*.cpp espejo de include/)
├── tools/
│   ├── bpm_cli.cpp               # exe: input.wav/.mp3 → output.json
│   └── dump_stage.cpp            # vuelca etapas intermedias a .bin (debug)
├── python_ref/
│   ├── gen_golden.py             # genera golden.npz (NUEVO, ver §3)
│   ├── generate_reference.py     # (del doc original)
│   └── compare_outputs.py        # métrica de similitud
└── tests/
    ├── golden/                   # golden.npz + golden_beats.json
    ├── test_01_windows.cpp
    ├── test_02_audio.cpp
    ├── test_03_stft_mel.cpp
    ├── test_04_onset.cpp
    ├── test_05_tempo.cpp
    ├── test_06_dp.cpp
    └── test_07_e2e.cpp
```

---

## 3. EL GOLDEN MASTER (fundamento de todo testing)

`python_ref/gen_golden.py` ejecuta el pipeline real de librosa sobre la canción y vuelca **todos** los intermedios a `golden.npz`. Esto desacopla "¿el algoritmo está bien?" de "¿la etapa N está bien?".

```python
# gen_golden.py — ESQUELETO
import numpy as np, librosa, json
y, sr = librosa.load("audio.wav",
                     sr=22050, mono=True)
# Etapa 1 intermedios
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512, window='hann',
                        center=True, pad_mode='constant'))
S_power = S**2
S_mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_fft=2048, hop_length=512,
                                       n_mels=128, fmax=sr/2, htk=False, norm='slaney')
S_db = librosa.power_to_db(S_mel)                       # ← ref=1, amin=1e-10, top_db=80
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
melfb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=128, fmax=sr/2, norm='slaney')
# Etapa 2
win_length = librosa.time_to_frames(8.0, sr=sr, hop_length=512)   # 344
tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                               hop_length=512, win_length=win_length)
tg_avg = np.mean(tg, axis=-1)                            # (344,)
bpms = librosa.tempo_frequencies(win_length, hop_length=512, sr=sr)
tempo = float(librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=512)[0])
# Etapa 3 (internos — vía API privada o re-implementación guiada)
tempo_bt, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

np.savez("golden.npz", y=y.astype(np.float32),
         S_power=S_power, S_mel=S_mel, S_db=S_db, oenv=oenv,
         melfb=melfb, tg=tg, tg_avg=tg_avg, bpms=bpms,
         tempo=np.float64(tempo), beats=beats, beat_times=beat_times,
         win_length=np.int64(win_length))
json.dump({"tempo":tempo,"beat_times":beat_times.tolist()}, open("golden_beats.json","w"))
```

Para `localscore`, `cumscore`, `backlink`, `FPB`: librosa los tiene como `@numba.guvectorize` privados.
Se re-implementan en numpy puro dentro de `gen_golden.py` **copiando el fuente** (lo tengo extraído) y se
vuelcan también, para tener el golden de la etapa 6.

---

## 4. PLAN POR FASES (con fórmulas CORRECTAS y pseudocódigo)

### 🔵 FASE 0 — Infraestructura (½ día)
- `CMakeLists.txt`: FetchContent KissFFT, dr_wav, dr_mp3, json, GoogleTest. C++17, `/O2`.
- `constants.h`: `SR=22050, HOP=512, N_FFT=2048, N_MELS=128, FMAX=11025, START_BPM=120, TIGHTNESS=100, MAX_TEMPO=320, AC_SIZE=8.0, AMIN=1e-10, TOP_DB=80`.
- **Test bloqueante:** `cmake --build` produce un exe vacío.

### 🔵 FASE 1 — Ventanas, Mel y matemáticas (1 día)
**Archivos:** `math_util.{h,cpp}`, `mel_filterbank.{h,cpp}`

```cpp
// Hann periódica (CORRECCIÓN C1): denominador = N
inline std::vector<double> hann_periodic(int N){
    std::vector<double> w(N);
    for(int n=0;n<N;++n) w[n]=0.5-0.5*std::cos(2*M_PI*n/N);
    return w;
}
// hanning(5) exacta de numpy (CORRECCIÓN C3)
inline std::array<double,5> hanning5(){ return {0,0.5,1.0,0.5,0}; }

// Mel Slaney (verificado)
double hz_to_mel(double f){ return f<1000.0 ? f/66.66666666666667
                                  : 15.0 + std::log(f/1000.0)/0.06875177742094912; }
double mel_to_hz(double m){ return m<15.0 ? m*66.66666666666667
                                  : 1000.0*std::exp(0.06875177742094912*(m-15.0)); }
// mediana de 128 (par): media de idx 63 y 64 tras ordenar
double median128(double v[128]){ std::nth_element(v,v+63,v+128);
                                 std::nth_element(v+64,v+64,v+128);
                                 return 0.5*(v[63]+v[64]); }   // ojo nth_element muta
```

**Melbank (CORRECCIÓN C4)** — fórmula del código real, no la del doc:
```cpp
// mel_f[0..129] = 130 frecuencias; fdiff[i]=mel_f[i+1]-mel_f[i]
// fftfreqs[k] = k*sr/n_fft
for(int i=0;i<N_MELS;++i){
  double fd0=mel_f[i+1]-mel_f[i], fd1=mel_f[i+2]-mel_f[i+1];
  for(int k=0;k<1+N_FFT/2;++k){
     double lower = (fftfreqs[k]-mel_f[i])/fd0;        // ascendente mel_f[i]→mel_f[i+1]
     double upper = (mel_f[i+2]-fftfreqs[k])/fd1;      // descendente mel_f[i+1]→mel_f[i+2]
     double v = std::max(0.0, std::min(lower,upper));
     H[i][k] = v * (2.0/(mel_f[i+2]-mel_f[i]));        // Slaney enorm
  }
}
```
**Tests:** `test_01` — `hann_periodic(2048)` y `(344)` bit-exactos vs golden; `hanning5()` bit-exacto;
`melfb` vs golden con max abs diff `< 1e-9` (ya verificado que es alcanzable).

### 🔵 FASE 2 — Audio loader + frame count (½ día)
**Archivos:** `audio_loader.{h,cpp}`
- `load_mono_resampled(path)` → `std::vector<double>` float64 @22050 mono.
- dr_wav: lee float32, promueve a float64, mezcla canales (media).
- dr_mp3: idem.
- **CORRECCIÓN C2:** número de frames se calcula al inicio: `int M = 1 + N/HOP;`

**Tests:** `test_02` — comparar `y` del WAV (no MP3) vs golden: MAE relativo `< 1e-7`.
⚠️ MP3 puede diferir — se valida aparte en §10.

### 🔵 FASE 3 — STFT + Mel + dB (1 día)
**Archivos:** `stft.{h,cpp}`, `power_to_db.{h,cpp}`

```cpp
// STFT: pad 1024 ceros cada lado, frame*2048*hann, RFFT→|·|²
int M = 1 + N/HOP;                                       // M_frames
Matrix<double> S_power(1+N_FFT/2, M, 0.0);              // 1025 × M_frames
auto w = hann_periodic(N_FFT);
kiss_fftr_cfg cfg = kiss_fftr_alloc(N_FFT, 0,0,0);
for(int m=0;m<M;++m){
   kiss_fft_scalar in[2048];
   for(int n=0;n<N_FFT;++n) in[n] = y_padded[m*HOP+n]*w[n];   // y_padded=pad1024
   kiss_fft_cpx out[1025];
   kiss_fftr(cfg, in, out);
   for(int k=0;k<=N_FFT/2;++k) S_power(k,m)=double(out[k].r*out[k].r+out[k].i*out[k].i);
}
```
```cpp
// Mel = H (128×1025) · S_power (1025×M)
Matrix<double> S_mel = gemm(H, S_power);                 // (128,M)
// power_to_db con top_db=80 (AÑADIDO A1)
double ref=1.0;
for cada elem:  log_spec = 10*log10(max(AMIN, x)) - 10*log10(max(AMIN, ref));
double mx = max(log_spec);
log_spec = max(log_spec, mx - TOP_DB);                   // clamp 80 dB
```
**Tests:** `test_03` — `S_mel[0:10,0:10]` MAE rel `<1e-5`; `S_db[0:10,0:10]` err abs `<0.01 dB`.

### 🔵 FASE 4 — Onset detector (½ día)
**Archivos:** `onset_detector.{h,cpp}`
```cpp
// flux con lag=1: flux[i,m]=max(0, S_db[i,m]-S_db[i,m-1])  para m=1..M-1
// agregación: oenv_raw[m-1] = median_128(flux[:,m])
// pad: 3 ceros (lag + n_fft/(2*hop) = 1+2) al inicio
// trim: oenv = oenv[0:M]   (ya coincide porque oenv_raw tiene M-1 → +3 = M+2 → [:M])
std::vector<double> oenv(M);
oenv.assign(M, 0.0);
for(int m=0; m < (M-1); ++m){                            // m frame destino en raw
   double col[128]; for(int i=0;i<128;++i) col[i]=std::max(0.0, S_db(i,m+1)-S_db(i,m));
   oenv_raw[m] = median128(col);
}
// padding: onset_env_final[t] = oenv_raw[t-3] si t-3∈[0,M-1) sino 0
```
**Tests:** `test_04` — `oenv` correlacion `>0.9999`, MAE rel `<1e-4`. **Esta es la metrica bisagra:** si pasa, Fases 1-4 correctas.

### 🔵 FASE 5 — Tempogram + Tempo (1 día)
**Archivos:** `tempogram.{h,cpp}`, `tempo_estimator.{h,cpp}`

```cpp
int WL=344;
auto acw = hann_periodic(WL);
// 1. pad oenv con linear_ramp de WL/2=172 a cada lado
std::vector<double> opad = linear_ramp_pad(oenv, 172);
// 2. frame hop=1, len=344 → odf_frame[t] = opad[t..t+343], para t=0..M-1
// 3. para cada columna t: x[τ]=opad[t+τ]*acw[τ]; AC por FFT (n_pad=720)
//    tg[:,t] = irfft(|rfft(x,n=720)|²)[0:344]
// 4. norm L∞ por columna (dividir por max|tg[:,t]|)   ← ANTES del mean
// 5. tg_avg[τ] = mean_t tg[τ,t]
```
```cpp
// prior + score
// bpms[0]=inf ; bpms[L]=2583.984375/L  para L=1..343
// logprior[L] = -0.5*((log2(bpms[L])-log2(120))/1.0)²
// max_idx = argmax(bpms < 320)            ← AÑADIDO A3
// logprior[0..max_idx-1] = -inf
// score[L] = log1p(1e6*tg_avg[L]) + logprior[L]
// best_lag = argmax score ; tempo = bpms[best_lag]
```
**Tests:** `test_05` — `tg_avg` correlacion `>0.999`; `tempo` dentro de `+-0.01 BPM` del golden.

### 🔵 FASE 6 — Beat tracker DP (1 día)  ⚠️ la fase más delicada
**Archivos:** `beat_tracker.{h,cpp}`

```cpp
double frame_rate = double(SR)/HOP;                     // 43.0664
int FPB = (int)std::llround(frame_rate*60.0/tempo);     // ~25

// 3.1 normalize onsets (ddof=1!)
double mu=mean(oenv), sd=stddd1(oenv);                  // std con N-1
double eps = TINY64;                                    // 2.225e-308
for(t) on[t] = oenv[t]/(sd+eps);

// 3.2 local score (gauss blur)
double sigma = FPB/32.0;                                // NOTA: window usa *32/FPB
std::vector<double> win(2*FPB+1);                       // k=-FPB..FPB
for(int k=-FPB;k<=FPB;++k) win[k+FPB]=exp(-0.5*(k*32.0/FPB)*(k*32.0/FPB));
localscore = same_convolve(on, win);                    // same mode

// 3.3 DP  (AÑADIDO A4: extremo inferior EXCLUSIVO)
double thresh = 0.01*max(localscore);
bool first=true;
backlink[0]=-1; cumscore[0]=localscore[0];
for(int i=1;i<M;++i){
   double best=-INFINITY; int bloc=-1;
   int lo = i - (int)std::llround(FPB/2.0);
   int hi = i - 2*FPB - 1;                              // exclusivo
   for(int loc=lo; loc>hi; --loc){                      // step -1
      if(loc<0) break;
      double sc = cumscore[loc] - TIGHTNESS*std::pow(std::log(i-loc)-std::log(FPB),2);
      if(sc>best){best=sc; bloc=loc;}
   }
   cumscore[i] = (bloc>=0)? localscore[i]+best : localscore[i];
   if(first && localscore[i] < thresh) backlink[i]=-1;
   else { backlink[i]=bloc; first=false; }
}
```
**Backtracking** (AÑADIDO A2: `localmax` asimétrico, `x[0]` nunca):
```cpp
auto lmax = localmax_asymmetric(cumscore);              // x[i]>x[i-1] && x[i]>=x[i+1]; x[0]=false
double med = median_of(cumscore where !lmax);           // masked median
double thr  = 0.5*med;
int tail=-1; for(int t=M-1;t>=0;--t) if(!lmax[t] && cumscore[t]>=thr){tail=t;break;}
std::vector<char> beats(M,0);
for(int t=tail; t>=0; t=backlink[t]) beats[t]=1;
```
**Trim** (CORRECCIÓN C3 + detalle `localscore[beats]`):
```cpp
auto w5 = hanning5();                                   // [0,0.5,1,0.5,0]
std::vector<double> boe; for(t) if(beats[t]) boe.push_back(localscore[t]);
auto sm = convolve(boe, w5);                            // luego slicing len(w)//2..
double rms = sqrt(mean(sm.*sm));  double thrT = 0.5*rms;
int n=0;   while(n<M && localscore[n]<=thrT){beats[n]=0;n++;}
int n=M-1; while(n>=0 && localscore[n]<=thrT){beats[n]=0;n--;}
```
**Tests:** `test_06` — `localscore` correlación `>0.999`; `backlink` idéntico en `≥99%` de frames;
mismo `tail`; `beats` boolean idéntico tras trim.

### 🔵 FASE 7 — Integración + CLI (½ día)
**Archivos:** `bpm_detector.{h,cpp}`, `beat_utils.{h,cpp}`, `tools/bpm_cli.cpp`
- API pública `BPMDetector::analyze_file()` (mismo header que §9 del doc).
- `frames_to_time`: `t = frame*HOP/SR`.
- `local_bpms`: `60/Δt` entre beats.
- CLI: `bpm_cli input output.json` con mismo formato que `reference_python.json`.

**Tests:** `test_07` — `similarity_score(cpp, golden, tol=0.04) ≥ 93%`; bonus tol=0.02.

---

## 5. PLAN DE TESTING — MATRIZ COMPLETA

| Test | Etapa | Golden campo | Tolerancia | Bloquea siguiente fase |
|------|-------|--------------|------------|------------------------|
| 01 | Ventanas/Mel | `hann(2048)`, `hann(344)`, `hanning5`, `melfb` | bit-exacto / `<1e-9` | Sí |
| 02 | Audio (WAV) | `y` | MAE rel `<1e-7` | Sí |
| 03 | STFT+Mel+dB | `S_mel[0:10,0:10]`, `S_db[0:10,0:10]` | `<1e-5` / `<0.01 dB` | Sí |
| 04 | Onset | `oenv` | corr `>0.9999`, MAE rel `<1e-4` | **Si (bisagra)** |
| 05 | Tempo | `tg_avg`, `tempo` | corr `>0.999` / `±0.01 BPM` | Sí |
| 06 | DP | `localscore`, `backlink`, `tail`, `beats` | corr `>0.999` / 99% idéntico | Sí |
| 07 | E2E | `beat_times` | `similarity ≥93%` (tol 0.04s) | — |

**Regla de oro del desarrollo:** ninguna fase se considera terminada hasta que su test pasa contra el golden. Si falla, **no avanzar**: depurar esa etapa con `dump_stage` (vuelca el buffer a `.bin`, se carga en numpy con `np.fromfile` y se compara contra golden).

---

## 6. ORDEN DE EJECUCIÓN Y DEPENDENCIAS

```
Fase0 ─┬─► Fase1 ─► Fase3 ─► Fase4 ─► Fase5 ─► Fase6 ─► Fase7 ─► E2E
       └─► Fase2 ───────────────────────────────────────► CLI
Gen golden ──────────────────────────────────────────────► (valida todo)
```
- `gen_golden.py` se ejecuta **primero** (es el contrato).
- Fases 1 y 2 pueden correr en paralelo (distintos archivos).
- Cada fase bloquea la siguiente: tolerancia estricta.

**Estimación total:** 4–5 días de desarrollo focalizado + 1–2 días de afinado MP3.

---

## 7. CRITERIOS DE ACEPTACIÓN (definición de "terminado")

1. ✅ `test_07` pasa: `similarity_score(cpp_beats, golden_beats, tol=0.04s) ≥ 93%`.
2. `tempo` dentro de `+-0.01 BPM` del golden.
3. ✅ Todos los tests 01–06 en verde (no se puede saltar).
4. Mismo numero de beats que Python `+-5%`.
5. ✅ CLI reproduce `reference_python.json` con mismo formato de campos.
6. ✅ Build limpio con `/W4` (o `-Wall -Wextra`) sin warnings.

---

## 8. RIESGOS Y MITIGACIONES

| Riesgo | Prob. | Impacto | Mitigación |
|--------|-------|---------|-----------|
| **R1.** dr_mp3 ≠ ffmpeg en samples | Alta | Alto | **Estrategia WAV-first:** validar todo contra WAV (bit-exacto). MP3 se mide aparte. Si <93%, volcar PCM crudo de Python y consumirlo en C++ (bybassa el decodificador). |
| **R2.** Divergencia numérica float64 en el DP | Media | Alto | Operar todo en float64; validar `backlink` frame-a-frame (no solo el score final). |
| **R3.** `argmax` con empates (-inf en prior) | Baja | Medio | `np.argmax` devuelve el **primer** máximo → replicar (primer L válido con score max). |
| **R4.** KissFFT vs scipy.fft precisión | Baja | Medio | Test con impulso unitario; differencias `<1e-12`. Si no, FFTW3f. |
| **R5.** `nth_element` para mediana muta el array | Baja | Bajo | Copiar `col` antes de ordenar. |
| **R6.** `frames_to_time` con división float | Baja | Bajo | Usar `double(512)/22050`, no `512/22050` (int). |

---

## 9. ENTREGABLES FINALES

1. Biblioteca `librosa_bpm` (headers + src) en `C:\Opencode\New folder\`.
2. Ejecutable `bpm_cli.exe`.
3. `golden.npz` + `golden_beats.json` regenerables.
4. Suite de tests 01–07 verde.
5. `README.md` con instrucciones de build y uso.
6. Documento `IMPLEMENTACION_NOTAS.md` registrando cualquier desviación de este plan.

---

## 10. NOTA SOBRE EL ORDEN DE VALIDACIÓN (lo que asegura llegar a 93%)

El mayor riesgo **no es el algoritmo** (está 100% especificado y verificado), sino el **decodificador de audio**.
Por eso el plan fuerza este orden:

1. **Primero WAV** (bit-exacto): prueba que Fases 1–7 están bien. Aquí el target es `≥95%` (margen).
2. **Después MP3** con dr_mp3: mide cuánto cae la similitud por el decodificador.
3. Si cae por debajo de 93%: consumir PCM pre-decodificado (R1) — garantiza el objetivo.

Esto desacopla los dos problemas y hace que el 93% sea **alcanzable de forma determinista**, no por suerte.
