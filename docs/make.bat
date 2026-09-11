@echo off
setlocal
pushd "%~dp0.."
if "%UV%"=="" set "UV=uv"
if "%1"=="html" goto build
if "%1"=="build" goto build
if "%1"=="serve" goto serve
if "%1"=="clean" goto clean
echo Usage: docs\make.bat [html^|build^|serve^|clean]
set "RESULT=1"
goto end
:build
%UV% run zensical build --strict --clean
goto result
:serve
%UV% run zensical serve
goto result
:clean
%UV% run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('site', '.cache', 'docs/_build')]"
:result
set "RESULT=%ERRORLEVEL%"
:end
popd
exit /b %RESULT%
