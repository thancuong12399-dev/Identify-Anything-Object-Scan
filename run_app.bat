@echo off
chcp 65001 > nul
title Dollar Street AI Studio - Phần Mềm Nhận Diện Vật Thể
echo ========================================================
echo   Đang khởi động phần mềm Dollar Street AI Studio...
echo ========================================================
cd /d "%~dp0"
python app.py
pause
