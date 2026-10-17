@echo off
rem Strata 启动器：将 strata.exe、web/ 与本脚本放在同一目录，然后运行。
rem 用法： start.bat [模型路径] [端口]
setlocal
set "DIR=%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%STRATA_MODEL%"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"

if "%MODEL%"=="" (
  echo 用法: start.bat ^<gguf 模型路径^> [端口]
  echo 或设置环境变量 STRATA_MODEL 后直接运行 start.bat
  exit /b 1
)

"%DIR%strata.exe" serve --model "%MODEL%" --web-dir "%DIR%web" --port %PORT%
endlocal
