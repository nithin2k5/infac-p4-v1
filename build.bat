@echo off
setlocal

echo ============================================================
echo  InFac P4 - Windows Build
echo ============================================================
echo.

:: ── Check venv exists ────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Run this first:
    echo   python -m venv venv
    echo   venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo   pip install pyinstaller
    pause
    exit /b 1
)

:: ── Activate venv ────────────────────────────────────────────
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: ── Check PyInstaller is available ───────────────────────────
where pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] PyInstaller not found in venv. Installing...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install PyInstaller.
        pause
        exit /b 1
    )
)

:: ── Clean previous build artefacts ───────────────────────────
echo [INFO] Cleaning previous build...
if exist "dist\InFacP4.exe" del /f /q "dist\InFacP4.exe"
if exist "build\InFacP4" rd /s /q "build\InFacP4"

:: ── Run PyInstaller ──────────────────────────────────────────
echo [INFO] Building InFacP4.exe — this may take a few minutes...
echo.
pyinstaller main.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed. Check the output above for details.
    pause
    exit /b 1
)

:: ── Done ─────────────────────────────────────────────────────
echo.
echo ============================================================
echo  Build complete!  dist\InFacP4.exe is ready.
echo ============================================================
echo.
echo NOTE: First launch may take 5-15 seconds while Windows
echo extracts the application to a temporary directory.
echo.
pause
endlocal
