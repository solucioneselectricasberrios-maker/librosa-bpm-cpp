# 🤖 GUÍA PARA AGENTES IA — "Porta librosa a C++"

> **Si eres un agente IA al que se le encomienda "portar librosa a C++" (o reanudar/
> auditar este port), LEE ESTO PRIMERO.** Contiene todo lo que necesitás saber para
> lograr exactamente lo mismo que ya está logrado aquí, sin repetir errores.

---

## 0. Lo primero que tenés que saber

**Este port YA ESTÁ HECHO Y VERIFICADO AL 100%.** Antes de escribir código nuevo,
verificá si lo que te piden ya existe. El estado actual:

- Pipeline completo de detección de BPM/beats en C++17 (7 etapas).
- **Similitud 100%** vs librosa 0.11.0 (objetivo era ≥93%).
- 7 tests CTest en verde, golden master completo.
- WAV-first bit-exacto; MP3 pendiente (ver §6).

Si tu tarea es **extender** (no rehacer), andá directo al §5.
Si tu tarea es **reproducir desde cero**, seguí el §1 al §4.

---

## 1. El error más caro que podés cometer: confiar en la documentación de librosa

**No leas solo la docstring de librosa. Leé el código fuente.** Durante este port se
encontraron **4 fórmulas erróneas** en el plan original que provenían de descripciones
imprecisas. Cada una se verificó ejecutando el código real de librosa, no leyendo su doc.

```python
# Para inspeccionar el fuente REAL (no la doc):
import librosa, inspect
print(inspect.getsource(librosa.beat.beat_track))
print(inspect.getsource(librosa.onset.onset_strength_multi))
# Los internos privados tienen name-mangling: getattr(librosa.beat, '__beat_track_dp')
```

**Regla:** si una fórmula del plan/librosa te da un resultado raro, ejecutala en Python
y comparala contra la salida de librosa antes de portarla. Los detalles están en
`docs/SECCION8_ANALISIS_CRITICO.md` y `docs/FORMULAS_VERIFICADAS.md`.

---

## 2. La metodología que garantiza el éxito (no la saltees)

El port se logró siguiendo **exactamente** este orden. Saltarse pasos lleva a depurar
C++ sin saber si el bug está en la fórmula o en la traducción.

### Paso 1 — Re-implementar en numpy puro ANTES de C++
Existe `python_ref/reference_impl.py`: el pipeline **entero** escrito solo con
numpy/scipy.fft, sin llamar a `librosa.*` (salvo para comparar). Esto demostró que las
fórmulas corrigen reproducen librosa al 100%.

**Por qué importa:** si tu C++ no cuadra, podés aislar el bug comparando contra
`reference_impl.py` (numpy) en vez de contra librosa (caja negra con numba).

### Paso 2 — Generar el golden master
`python_ref/gen_golden.py` ejecuta librosa real y vuelca **TODOS** los intermedios
a `golden.npz`: `y, S_power, S_mel, S_dB, oenv, melfb, tg, tg_avg, tempo, localscore,
cumscore, backlink, beats`. Esto es el contrato.

```python
import python_ref.gen_golden as G
# Los internos del beat tracker se re-implementan dentro de gen_golden.py porque
# librosa los tiene como @numba.guvectorize privados. Copiar de ahí.
```

### Paso 3 — Volcar el golden a formato que C++ pueda leer sin dependencias
`python_ref/export_golden_bin.py` convierte cada array `.npz` a `.bin` crudo
(little-endian, C-contiguous) + un `golden_manifest.json` con shapes/dtypes.
Los tests C++ leen esto con `fopen`/`fread` (sin cnpy/zlib).

### Paso 4 — Portar etapa por etapa, test por test
Cada etapa tiene un test que la bloquea. **No avances hasta que el test pase.**
Orden: ventanas → audio → STFT/mel → onset → tempo → DP → e2e.

---

## 3. Los 6 detalles que rompen la similitud (memorizalos)

Estos son los puntos donde TODO port falla. Están verificados empíricamente.

### 3.1 Ventana Hann periódica → denominador **N**
```cpp
// BIEN:  w[n] = 0.5 - 0.5*cos(2*pi*n/N)
// MAL:   w[n] = 0.5 - 0.5*cos(2*pi*n/(N+1))   <- error típico
// MAL:   w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))   <- Hann simétrica
```
Para STFT `N=2048`, para tempogram `N=344`.

### 3.2 Número de frames → `M = 1 + N//hop`
Con `center=True, pad_mode='constant'`, librosa rellena `n_fft//2=1024` ceros a **cada
lado** y luego enmarca. Simplificado: `M = 1 + N//hop` (N = muestras de audio).
NO uses `ceil((N+1024)/hop)` ni `(N+2047)//hop+1`.

### 3.3 Melbank → `lower` ascendente, `upper` descendente
```cpp
// Fórmula del CÓDIGO REAL de librosa (no la de la doc, que da suma 0):
for (int k...) {
    double lower = (fftfreqs[k] - mel_f[i])     / (mel_f[i+1] - mel_f[i]);     // sube 0->1
    double upper = (mel_f[i+2] - fftfreqs[k])   / (mel_f[i+2] - mel_f[i+1]);   // baja 1->0
    H[i][k] = max(0, min(lower, upper)) * (2.0 / (mel_f[i+2] - mel_f[i]));     // enorm Slaney
}
```

### 3.4 `power_to_db` con `top_db=80` ACTIVO
El plan original no lo mencionaba. `librosa.power_to_db` clampa el rango dinámico a 80 dB
por debajo del pico. Sin esto, el spectral flux cambia y el onset se degrada.

### 3.5 `np.hanning(5)` = `[0, 0.5, 1.0, 0.5, 0]`
Usada en el trim de beats. NO es `[0, 0.345, 0.655, 0.345, 0]` (esa sería Hann periódica
mal aplicada).

### 3.6 `localmax` **asimétrico** + el rango del DP
```cpp
// localmax: x[0] NUNCA es máximo. x[-1] lo es si x[-1] > x[-2].
m[i] = (x[i] > x[i-1]) && (x[i] >= x[i+1]);   // i=1..n-2

// Rango del bucle de predecesores del DP (extremo inferior EXCLUSIVO):
for (int loc = i - round(FPB/2); loc > i - 2*FPB - 1; --loc) { ... }
//                                                            ^^^ -1 = exclusivo
```

---

## 4. Cómo reproducir este port desde cero (receta paso a paso)

Si te piden rehacerlo (por ejemplo en otro workspace), este es el orden probado:

1. **Verificá el toolchain.** Windows + MSVC + CMake. Probá compilar un hello-world
   cargando `vcvars64.bat` (ver `build.bat`).
2. **Instalá las deps Python:** `pip install numpy scipy "librosa==0.11.0" soundfile`.
3. **Generá el golden:** `python python_ref/gen_golden.py` (necesita el MP3 de prueba).
4. **Validá el reference_impl:** `python python_ref/validate_reference_impl.py` → 15/15 PASS.
5. **Exportá .bin:** `python python_ref/export_golden_bin.py`.
6. **Copiá third_party/** (kissfft, dr_wav.h, dr_mp3.h, nlohmann/json.hpp) — ya están incluidos.
7. **Portá etapa por etapa** siguiendo `docs/PLAN_IMPLEMENTACION_CPP.md` y los tests 01→07.
8. **Cada test debe pasar antes de avanzar.** Si uno falla, usá `reference_impl.py` para aislar.

---

## 5. Cómo extender este port (tareas comunes)

### "Añadir soporte MP3 con dr_mp3"
`third_party/dr_mp3.h` ya está incluido. Falta implementar `load_mp3()` en
`src/audio_loader.cpp` (instantiating `DR_MP3_IMPLEMENTATION`). Ver §6 abajo: el riesgo
es la divergencia de samples vs ffmpeg.

### "Añadir resampling (cargar MP3 a otra sr)"
El golden asume audio ya a 22050 Hz mono. Para sr arbitraria necesitás un resampler
(lineal o libsamplerate). El WAV de prueba ya está a 22050, así que los tests no lo ejercitan.

### "Añadir onset_detect / tempo dinámico"
El tempogram completo (`tg` 344×M) ya se computa en `tempogram.h`. Para tempo dinámico,
usá `aggregate=None` en el eje temporal (no el `mean`). Está fuera del scope actual.

### "Cambiar parámetros (hop_length, n_mels, etc.)"
Editá `include/librosa_bpm/constants.h`. **Regenerá el golden** con los nuevos parámetros
en `gen_golden.py` antes de testear (cambian M, win_length, etc.).

### "Portar a otro SO / compilador"
El código es C++17 estándar. En Linux/Mac: reemplazá `build.bat` por `cmake -B build &&
cmake --build build`. KissFFT y dr_libs son multiplataforma. Solo `_CRT_SECURE_NO_WARNINGS`
y `fopen_s` (en `audio_loader.cpp`) son específicos de MSVC — usá `#ifdef _MSC_VER`.

---

## 6. El riesgo MP3 (importante)

`librosa.load` decodifica MP3 vía **ffmpeg**. `dr_mp3` (lo que usaría el C++) **no da
samples bit-iguales**. Las diferencias (~1 LSB en float32) se acumulan en el STFT y pueden
desplazar beats.

**Por eso este port se valida WAV-first:**
- `tests/golden/song_f32.wav` es el WAV 32-bit float de la canción (producido por soundfile,
  el mismo backend que usa librosa para WAV). `dr_wav` lo lee **bit-exacto**.
- Aquí el C++ da 100% de similitud.

**Si necesitás validar MP3:**
1. Implementá `load_mp3()` con `dr_mp3.h`.
2. Medí la similitud del MP3 contra el golden WAV. Si cae por debajo del objetivo:
3. **Mitigación:** decodificá el MP3 una vez con Python (`librosa.load`), volcá el PCM crudo
   a `.pcm` (`song_f32.pcm` ya existe como ejemplo) y consumí eso en C++ con `load_pcm_f32()`.
   Esto bypasea el decodificador y garantiza bit-exactitud.

---

## 7. Cómo depurar cuando algo no cuadra

El error típico: un test falla y no sabés si es la fórmula o el C++.

1. **Aislar con reference_impl.py.** El test C++ compara contra `.bin` (de librosa).
   Pero también podés comparar tu implementación numpy de la etapa sospechosa contra
   `reference_impl.py` (que ya está validado al 100%).
2. **Volcar el buffer C++.** Añadí un `dump_stage` que escriba tu array intermedio a `.bin`.
   Cargalo en Python: `np.fromfile("mibuffer.bin", dtype=np.float64)` y compará con golden.
3. **Comparar por etapas, no de golpe.** Si `oenv` cuadra pero `tempo` no → el bug es del
   tempogram. Si `tempo` cuadra pero beats no → el bug es del DP. La matriz de tests del
   README mapea cada etapa a su tolerancia.

---

## 8. Archivos que NUNCA debes borrar o modificar sin regenerar

- `tests/golden/golden.npz` y `tests/golden/bin/*` → son el contrato. Si cambiás parámetros,
  **regenerá** con `gen_golden.py` + `export_golden_bin.py`.
- `python_ref/reference_impl.py` → es la fuente de verdad de las fórmulas. Cualquier cambio
  en C++ debe reflejarse aquí primero y validarse con `validate_reference_impl.py`.
- `third_party/` → no modificable (son libs externas).

---

## 9. Comando de smoke test (validá que todo sigue funcionando)

```bat
cd Desktop\LIBROSACplusplus
build.bat test
build\Release\bpm_cli.exe tests\golden\song_f32.wav tests\golden\output_cpp.json
python python_ref\compare_outputs.py tests\golden\golden_beats.json tests\golden\output_cpp.json
```
Ambos deben dar PASS. Si no, algo se rompió y hay que investigar antes de cualquier cambio.
