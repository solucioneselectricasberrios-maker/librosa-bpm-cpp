@echo off
REM build.bat - configura, compila y (opcional) testa el proyecto con MSVC + CMake
REM Uso: build.bat         -> configure + build
REM      build.bat test     -> + ctest
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

%CMAKE% -S . -B build -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 ( echo [build] ERROR configure & exit /b 1 )

%CMAKE% --build build --config Release
if errorlevel 1 ( echo [build] ERROR build & exit /b 1 )

if /I "%1"=="test" (
    echo.
    echo === ctest ===
    cd build
    ctest -C Release --output-on-failure
    cd ..
)
