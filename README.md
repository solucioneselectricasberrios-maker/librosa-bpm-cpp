# 🎵 Port de librosa beat detection a C++ — Documentación maestra

Este repositorio contiene un **port fiel y verificado** del pipeline de detección de
BPM/beats de `librosa` (Python) a **C++17 puro**, logrando **100% de similitud** en las
posiciones de beats respecto a la salida de referencia de librosa 0.11.0
(objetivo original: ≥93%).

> **Punto de entrada para cualquier agente IA o humano** que deba entender, reproducir,
> extender o mantener este port. Todo lo necesario está aquí: código, tests, golden master,
> toolchain, fórmulas exactas y la traza completa de las decisiones.

---

## 📊 Resultado verificado

| Métrica | Python (librosa) | C++ (este port) | Estado |
|---------|------------------|-----------------|--------|
| Tempo global | 117.4538 BPM | 117.4538 BPM | idéntico (Δ=0) |
| Nº de beats | 59 | 59 | idéntico |
| Onset envelope | — | correlación 1.000000 | idéntico |

**Audio de prueba:** click track sintetico a 120 BPM — 661,500 muestras @ 22,050 Hz (30.0s)

---

## 📁 Estructura del repositorio

```
LIBROSACplusplus/
├── README.md                          ← ESTE ARCHIVO (punto de entrada)
├── docs/                              ← Documentación técnica profunda
│   ├── SECCION8_ANALISIS_CRITICO.md       (errores encontrados en el plan original)
│   ├── PLAN_IMPLEMENTACION_CPP.md         (plan fase por fase con fórmulas)
│   ├── AGENT_GUIDE.md                     (cómo continuar/extendir — ver abajo)
│   └── FORMULAS_VERIFICADAS.md            (tabla de fórmulas exactas verificadas)
├── CMakeLists.txt                     ← Build (MSVC + KissFFT + dr_libs + json, todo local)
├── build.bat                          ← Script de build+test para Windows
├── include/librosa_bpm/               ← Librería (headers)
│   ├── constants.h                        Constantes del pipeline
│   ├── matrix.h                           Contenedor 2D row-major + matmul
│   ├── math_util.h                        Hann, Mel, median, std, convoluciones
│   ├── audio_loader.h                     API de carga (dr_wav, PCM)
│   ├── stft.h                             STFT con KissFFT (RFFT + |·|²)
│   ├── mel_filterbank.h                   Filtros Mel Slaney
│   ├── power_to_db.h                      dB con top_db=80
│   ├── onset_detector.h                   Onset strength envelope
│   ├── tempogram.h                        Autocorrelación + tempo
│   ├── beat_tracker.h                     DP de Ellis 2007
│   └── bpm_detector.h                     API pública (orquesta todo)
├── src/
│   └── audio_loader.cpp                   Implementación (instancia dr_wav aquí)
├── tools/
│   ├── bpm_cli.cpp                        exe: input.wav/.pcm -> output.json
│   └── smoke_test.cpp                     valida toolchain
├── tests/                             ← 7 tests CTest, todos verde
│   ├── golden.h                           lector de golden .bin + métricas
│   ├── test_01_windows_mel.cpp            ventanas + mel filterbank
│   ├── test_02_audio.cpp                  audio loader (WAV + PCM)
│   ├── test_03_stft_mel.cpp               STFT + Mel + dB
│   ├── test_04_onset.cpp                  onset envelope (métrica bisagra)
│   ├── test_05_tempo.cpp                  tempogram + tempo
│   ├── test_06_dp.cpp                     DP beat tracker
│   ├── test_07_e2e.cpp                    end-to-end con similitud
│   └── golden/                            ← golden master (NO borrar)
│       ├── golden.npz                     arrays numpy (referencia librosa)
│       ├── golden_beats.json              beats de referencia
│       ├── bin/                           arrays volcados a .bin + manifest
│       ├── song_f32.wav                   audio sintetico (click track)
│       ├── song_f32.pcm                   PCM crudo float32 (bypassea decodificador)
├── python_ref/                        ← Scripts Python (contrato + validación)
│   ├── gen_golden.py                      GENERA el golden master desde librosa
│   ├── reference_impl.py                  pipeline re-impl. en numpy puro (¡clave!)
│   ├── validate_reference_impl.py         valida reference_impl vs golden (15 checks)
│   ├── export_golden_bin.py               vuelca .npz a .bin para C++
│   └── compare_outputs.py                 métrica de similitud C++ vs Python
└── third_party/                       ← Dependencias (locales, sin descargas en build)
    ├── kissfft/                           FFT (kiss_fft.c, kiss_fftr.c)
    ├── dr_mp3.h                           decodificador MP3 (single-header)
    ├── dr_wav.h                           decodificador WAV (single-header)
    └── nlohmann/json.hpp                  JSON single-header
```

---

## 🚀 Cómo usarlo (guía rápida)

### Requisitos
- **Windows + Visual Studio 2022 Build Tools** (MSVC `cl.exe` + CMake). Verificado con MSVC 19.44.
- **Python 3.10+** con `numpy`, `scipy`, `librosa==0.11.0`, `soundfile` (solo para regenerar el golden).

### Build + test (lo más rápido)
```bat
cd Desktop\LIBROSACplusplus
build.bat test
```
Esto carga `vcvars64.bat`, configura con CMake, compila y ejecuta los 7 tests CTest.
Resultado esperado: **`100% tests passed, 0 tests failed out of 7`**.

### Usar el CLI
```bat
build\Release\bpm_cli.exe tests\golden\song_f32.wav salida.json
```
Produce un JSON con `global_tempo`, `beat_times`, `local_bpms`, `n_beats`.

### Comparar contra Python
```bat
python python_ref\compare_outputs.py tests\golden\golden_beats.json salida.json
```
Salida esperada: `Overall: 100.0%  PASS: True`.

---

## 🔑 Los 6 hallazgos críticos (lo que fallaba en el plan original)

El plan original (`LIBROSA_BPM_PORT_CPP.md`) tenía **4 fórmulas erróneas** y **omitía 2 detalles**
que habrían hecho imposible alcanzar el objetivo. Cada uno se verificó contra el código
fuente real de librosa. Detalle completo en `docs/SECCION8_ANALISIS_CRITICO.md`.

| # | Decía el plan original | Correcto (verificado) | Impacto |
|---|------------------------|----------------------|---------|
| 1 | Hann: `0.5-0.5·cos(2π·n/(N+1))` | `0.5-0.5·cos(2π·n/N)` | total |
| 2 | `M = ceil((N+1024)/512)` | `M = 1 + N//hop` | total |
| 3 | Trim window `[0,0.345,0.655,0.345,0]` | `np.hanning(5)=[0,0.5,1,0.5,0]` | medio |
| 4 | Melbank: A/B invertidas (suma=0) | `lower=(f-f₀)/(f₁-f₀)`, `upper=(f₂-f)/(f₂-f₁)` | total |
| 5 | (omitido) | `power_to_db` con `top_db=80` ACTIVO | alto |
| 6 | (omitido) | `localmax` **asimétrico**: `x[i]>x[i-1]` estr., `x[i]>=x[i+1]`, `x[0]` nunca | alto |

---

## 🧪 Filosofía de verificación: golden-master testing

El port se validó **etapa por etapa** contra un "golden master" generado con librosa real.
Cada una de las 7 etapas tiene su test, y **ninguna etapa se consideró terminada hasta
que su test pasó**. Esto desacopla "¿el algoritmo está bien?" de "¿la etapa N está bien?".

La pieza clave de la validación es **`python_ref/reference_impl.py`**: una re-implementación
**del pipeline entero en numpy puro** (sin llamar a `librosa.stft`, `melspectrogram`, etc.).
Antes de escribir una sola línea de C++, se demostró que esas fórmulas corrigen reproducen
librosa al 100%. Luego el C++ fue **traducción mecánica** de ese reference_impl.

```bat
REM Demostración de que las fórmulas son correctas (antes de C++):
python python_ref\validate_reference_impl.py
REM Resultado: 15 PASS / 0 FAIL, similarity 100.00%
```

---

## 📚 Documentación adicional (en `docs/`)

| Documento | Para qué sirve |
|-----------|----------------|
| `AGENT_GUIDE.md` | **Lo que debe leer un agente IA** al que se le enciende "porta librosa a C++" |
| `FORMULAS_VERIFICADAS.md` | Tabla de TODAS las fórmulas con su valor numérico verificado |
| `SECCION8_ANALISIS_CRITICO.md` | Análisis crítico de los errores del plan original |
| `PLAN_IMPLEMENTACION_CPP.md` | Plan fase por fase con pseudocódigo C++ |

---

## ⚠️ Estado y limitaciones

- ✅ **WAV-first validado al 100%**: el pipeline completo es bit-exacto contra librosa
  cuando el audio se carga como WAV (que es como librosa/soundfile lo procesa).
- ⚠️ **MP3 pendiente de validar**: `dr_mp3` ≠ `ffmpeg` en los samples decodificados, lo que
  puede degradar la similitud. Ver `docs/AGENT_GUIDE.md` § "Riesgo MP3" para la mitigación.
- 📌 El port cubre **únicamente** detección de BPM/beats (las 3 llamadas de `tempo_analysis.py`).
  NO incluye metrónomo, time-stretching ni UI.

---

## 📜 Licencia

Este proyecto se distribuye bajo los terminos de la **GNU General Public License v3.0 (GPL-3.0)**.
Ver el archivo [LICENSE](LICENSE) para el texto completo.

> **Nota sobre third_party/:** Los componentes en `third_party/` (KissFFT, dr_libs, nlohmann/json)
> tienen sus propias licencias (BSD, dominio publico/unlicense, MIT respectivamente) y no se ven
> afectados por la GPL de este proyecto.
