# Robot Voice Commander
Sistema de control por voz para brazo robótico ABB en ROS2 Jazzy.

```
Voz → Whisper STT → llama.cpp LLM → Action Parser → ROS2 topic → MoveIt2 → ABB
```

Todo corre **100% local**, sin internet, compatible con CPU y GPU NVIDIA.

---

## Requisitos del sistema

| Componente | Versión |
|---|---|
| Ubuntu | 24.04 |
| ROS2 | Jazzy |
| Python | 3.12 |
| CUDA | 12.1+ (solo si tienes GPU NVIDIA) |

---

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd voice_to_abb_ws
```

### 2. Crear entorno virtual

```bash
python3 -m venv voice_env
source voice_env/bin/activate
```

### 3. Instalar dependencias Python

**Si tienes CPU solamente (sin GPU):**
```bash
pip install pydantic pyyaml rich numpy faster-whisper ollama pyaudio silero-vad
pip install llama-cpp-python
```

**Si tienes GPU NVIDIA (CUDA):**
```bash
pip install pydantic pyyaml rich numpy faster-whisper ollama pyaudio silero-vad
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

> Para verificar si tienes CUDA: `nvidia-smi`

### 4. Instalar dependencias del sistema

```bash
sudo apt install portaudio19-dev python3-dev ffmpeg
```

### 5. Descargar el modelo LLM

El modelo pesa ~2GB y **no está en el repo**. Descárgalo una sola vez:

```bash
mkdir -p models
curl -L "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
     -o models/llama3.2-3b-q4.gguf \
     --progress-bar
```

> El modelo de Whisper se descarga automáticamente la primera vez que corres el sistema (~150MB).

### 6. Compilar el workspace ROS2

```bash
source /opt/ros/jazzy/setup.bash
colcon build
```

### 7. Configurar el entorno

Edita `setup_env.sh` y verifica que la ruta del workspace sea correcta. Luego:

```bash
source setup_env.sh
```

> De ahora en adelante, cada vez que abras una terminal nueva ejecuta este comando antes de usar el sistema.

---

## Configuración

El archivo principal es `src/robot_voice_commander/config/settings.yaml`.

Las opciones más importantes:

```yaml
whisper:
  model_size: "small"   # tiny | base | small — más grande = más preciso pero más lento
  language: "es"        # es | en | null (auto-detect)

llama_cpp:
  model_path: "/ruta/absoluta/a/models/llama3.2-3b-q4.gguf"
  n_threads: 4          # aumenta si tienes más núcleos de CPU
```

> **Importante:** ajusta `model_path` con tu ruta absoluta. Puedes obtenerla con `realpath models/llama3.2-3b-q4.gguf`

---

## Uso

### Correr el sistema completo

Abre dos terminales. En ambas ejecuta primero:
```bash
source setup_env.sh
```

**Terminal 1 — nodo principal:**
```bash
ros2 run robot_voice_commander voice_commander
```

**Terminal 2 — ver comandos publicados:**
```bash
ros2 topic echo /robot/voice_commands
```

Cuando veas `Esperando comando de voz...` habla claramente.

### Comandos de voz de ejemplo

| Lo que dices | Acción generada |
|---|---|
| "mueve el joint 1 a 45 grados" | `move_joint(joint1, 45°)` |
| "ve a home" | `move_home()` |
| "abre el gripper" | `open_gripper()` |
| "cierra el gripper" | `close_gripper()` |
| "para" / "stop" | `stop()` |
| "muévete 10 centímetros en X" | `move_cartesian(x=0.1)` |
| "rota el joint 2 treinta grados" | `rotate_joint(joint2, +30°)` |

---

## Benchmarks

Para reproducir los benchmarks del proyecto:

**Benchmark Whisper × Ollama** (compara modelos):
```bash
ros2 run robot_voice_commander benchmark
```

**Benchmark Ollama vs llama.cpp** (compara backends):
```bash
ros2 run robot_voice_commander benchmark_llama
```

Los resultados se guardan en `benchmark_resultados.json` y `benchmark_llama_cpp.json`.

### Resultados obtenidos (CPU, sin GPU)

| Whisper | LLM Backend | Precisión | T-Total |
|---|---|---|---|
| small | llama.cpp | **75%** | 19.2s |
| small | Ollama | 62% | 16.9s |
| tiny | llama.cpp | 62% | — |

> Con GPU NVIDIA el tiempo total estimado es ~1.5-2s por ciclo.

---

## Arquitectura

```
voice_to_abb_ws/
├── src/
│   ├── robot_voice_commander/        # Paquete principal
│   │   ├── config/
│   │   │   └── settings.yaml         # Configuración general
│   │   └── robot_voice_commander/
│   │       ├── modules/
│   │       │   ├── audio/            # Captura de micrófono + Whisper STT
│   │       │   ├── llm/              # Clientes Ollama y llama.cpp
│   │       │   ├── parser/           # Validación de comandos (Pydantic)
│   │       │   ├── context/          # Construcción del prompt
│   │       │   └── hardware.py       # Autodetección GPU/CPU
│   │       ├── pipeline.py           # Orquestador del pipeline
│   │       ├── voice_commander_node.py   # Nodo ROS2 principal
│   │       ├── moveit_executor_node.py   # Nodo ROS2 MoveIt2 (en desarrollo)
│   │       ├── benchmark.py          # Benchmark Whisper × Ollama
│   │       └── benchmark_cpp.py      # Benchmark Ollama vs llama.cpp
│   └── robot_voice_msgs/             # Mensajes ROS2 custom
│       └── msg/
│           ├── RobotCommand.msg
│           ├── RobotAction.msg
│           └── PipelineStatus.msg
├── models/                           # Modelos GGUF (no en git, descargar manualmente)
├── setup_env.sh                      # Script de entorno
└── README.md
```

---

## Topics ROS2

| Topic | Tipo | Descripción |
|---|---|---|
| `/robot/voice_commands` | `robot_voice_msgs/RobotCommand` | Comandos parseados listos para ejecutar |
| `/robot/pipeline_status` | `robot_voice_msgs/PipelineStatus` | Estado de cada ciclo del pipeline |

---

## Solución de problemas

**"No se detectó voz"**
Verifica tu micrófono con:
```bash
python3 -c "import pyaudio; pa = pyaudio.PyAudio(); [print(f'[{i}]', pa.get_device_info_by_index(i)['name']) for i in range(pa.get_device_count()) if pa.get_device_info_by_index(i)['maxInputChannels'] > 0]"
```
Ajusta `device_index` en `settings.yaml` con el índice correcto.

**"KeyError: model_path"**
Verifica que `settings.yaml` tiene la sección `llama_cpp` con la ruta correcta al modelo `.gguf`.

**Whisper lento**
Cambia en `settings.yaml`:
```yaml
whisper:
  model_size: "tiny"
```

**llama-cpp-python no usa GPU**
Reinstala con soporte CUDA:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
```

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `RVC_CONFIG` | `config/settings.yaml` | Ruta al archivo de configuración |