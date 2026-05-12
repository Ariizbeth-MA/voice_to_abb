#!/bin/bash
source /home/arizbeth/8_semestre/voice_to_abb_ws/voice_env/bin/activate
source /home/arizbeth/8_semestre/voice_to_abb_ws/install/setup.bash
export PYTHONPATH=/home/arizbeth/8_semestre/voice_to_abb_ws/voice_env/lib/python3.12/site-packages:$PYTHONPATH
export RVC_CONFIG=src/robot_voice_commander/config/settings.yaml
echo "Entorno listo"
