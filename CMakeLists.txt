cmake_minimum_required(VERSION 3.20)
project(librosa_bpm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)

if(MSVC)
    add_compile_options(/EHsc)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# ============================================================================
# Dependencias (todas locales en third_party/)
# ============================================================================

# --- KissFFT (real FFT, en double) ---
add_library(kissfft STATIC
    third_party/kissfft/kiss_fft.c
    third_party/kissfft/kiss_fftr.c
    third_party/kissfft/kfc.c
)
target_include_directories(kissfft PUBLIC third_party/kissfft)
target_compile_definitions(kissfft PUBLIC kiss_fft_scalar=double)
if(MSVC)
    target_compile_options(kissfft PRIVATE /W0)
endif()

# --- nlohmann/json + dr_libs (header-only, INTERFACE) ---
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE third_party)

add_library(dr_libs INTERFACE)
target_include_directories(dr_libs INTERFACE third_party)

# ============================================================================
# Libreria principal
# ============================================================================
add_library(librosa_bpm STATIC
    src/audio_loader.cpp
)
target_include_directories(librosa_bpm PUBLIC include)
target_link_libraries(librosa_bpm PUBLIC kissfft nlohmann_json)
# dr_libs solo lo necesita audio_loader.cpp (definiciones del single-header)
target_include_directories(librosa_bpm PRIVATE third_party)

# ============================================================================
# Tests (cada test es un exe que lee tests/golden/bin/)
# ============================================================================
enable_testing()

function(lbpm_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE librosa_bpm)
    target_include_directories(${name} PRIVATE tests)
    # ctest debe correr desde la raiz del proyecto para que "tests/golden/bin" sea visible
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

lbpm_test(test_01_windows_mel)
lbpm_test(test_02_audio)
lbpm_test(test_03_stft_mel)
lbpm_test(test_04_onset)
lbpm_test(test_05_tempo)
lbpm_test(test_06_dp)
lbpm_test(test_07_e2e)

# --- CLI ---
add_executable(bpm_cli tools/bpm_cli.cpp)
target_link_libraries(bpm_cli PRIVATE librosa_bpm)

# --- Smoke test (valida toolchain + json) ---
add_executable(smoke_test tools/smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE librosa_bpm)
