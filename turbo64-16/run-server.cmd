@echo off
setlocal
cd /d "%~dp0.."
set "TURBO64_16=2"
if not "%~1"=="" set "TURBO64_16=%~1"
if not "%TURBO64_16%"=="0" if not "%TURBO64_16%"=="1" if not "%TURBO64_16%"=="2" exit /b 2
echo Turbo64-16 mode %TURBO64_16%: 0=original, 1=mapped experts and read-ahead, 2=add pinned uploads
"turbo64-16\bin\llama-server.exe" ^
 -m "C:\Users\Tudi\Documents\Tudi\AI\MiMo-V2.6-Flash-MOPD-Q2_K-00001-of-00002.gguf" ^
 --host 0.0.0.0 --port 5559 ^
 -c 65536 -ngl all --n-cpu-moe 47 ^
 -t 16 -tb 24 -b 4096 -ub 1024 ^
 -fa on -lm mmap -np 1 --jinja --reasoning on
