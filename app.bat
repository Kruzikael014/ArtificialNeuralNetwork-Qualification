@echo off

if not exist "./model/emotion_model.ckpt.index" (
  echo Checkpoint file not found. Running main.py...
  python ./scripts/train.py
)

echo Running face_recognition.py...
python ./scripts/main.py

pause