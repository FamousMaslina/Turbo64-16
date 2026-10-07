@echo off
setlocal
cd /d "%~dp0.."
if "%~1"=="--turbo-prefill" goto profile
if "%~1"=="--turbo-decode" goto profile
if "%~1"=="--turbo-balanced" goto profile
set "TURBO64_16=2"
if not "%~1"=="" set "TURBO64_16=%~1"
set "TURBO_SERVER=turbo64-16\bin\llama-server.exe"
set "TURBO_MICROBATCH=1024"
if "%~1"=="baseline" (
 set "TURBO64_16=2"
 set "TURBO_SERVER=turbo64-16\baseline-log-build\llama-server.exe"
 set "TURBO64_16_CPU_PREFETCH=0"
)
if not "%~2"=="" set "TURBO_MICROBATCH=%~2"
if not "%TURBO_MICROBATCH%"=="1024" if not "%TURBO_MICROBATCH%"=="2048" if not "%TURBO_MICROBATCH%"=="4096" exit /b 2
if not "%TURBO64_16%"=="0" if not "%TURBO64_16%"=="1" if not "%TURBO64_16%"=="2" exit /b 2
if not exist "%TURBO_SERVER%" exit /b 2
echo Turbo64-16 mode %TURBO64_16%: 0=original, 1=mapped experts and read-ahead, 2=add pinned uploads
echo Microbatch %TURBO_MICROBATCH%, executable %TURBO_SERVER%
"%TURBO_SERVER%" ^
 -m "C:\Users\Tudi\Documents\Tudi\AI\MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf" ^
 --host 0.0.0.0 --port 5559 ^
 -c 65536 -ngl all --n-cpu-moe 47 ^
 -t 16 -tb 24 -b 4096 -ub %TURBO_MICROBATCH% ^
 -fa on -lm mmap -np 1 --jinja --reasoning on
exit /b %errorlevel%

:profile
set "TURBO64_16=2"
"turbo64-16\bin\llama-server.exe" %*
exit /b %errorlevel%
