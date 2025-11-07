@echo off
ECHO Starting all services...

REM This assumes you are running this .bat file from your main project folder

start "Backend" cmd /k "cd backend &&  node server.js"
start "Python Service" cmd /k "cd python_service && python main.py"
start "WSS" cmd /k "cd wss && npm start"