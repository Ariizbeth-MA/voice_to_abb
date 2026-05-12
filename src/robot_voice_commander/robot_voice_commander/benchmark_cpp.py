"""
Benchmark Ollama vs llama.cpp con Whisper small.
Compara latencia y precision para el reporte final.
"""

from __future__ import annotations

import json
import time
import yaml
import os
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .modules.audio import AudioCapture, Transcriber
from .modules.llm import LLMClient, LlamaCppClient
from .modules.parser import ActionParser
from .modules.context import ContextBuilder

console = Console()

COMANDOS = [
    {"esperado": "move_home",      "di": "ve a home"},
    {"esperado": "move_joint",     "di": "mueve el joint 1 a 45 grados"},
    {"esperado": "move_joint",     "di": "mueve el joint 3 a 90 grados"},
    {"esperado": "open_gripper",   "di": "abre el gripper"},
    {"esperado": "close_gripper",  "di": "cierra el gripper"},
    {"esperado": "stop",           "di": "para"},
    {"esperado": "move_cartesian", "di": "muevete 10 centimetros en X"},
    {"esperado": "rotate_joint",   "di": "rota el joint 2 treinta grados"},
]


@dataclass
class ResultadoCiclo:
    backend: str
    comando_esperado: str
    texto_transcrito: str = ""
    accion_detectada: str = ""
    correcto: bool = False
    t_audio: float = 0.0
    t_whisper: float = 0.0
    t_llm: float = 0.0
    t_total: float = 0.0


@dataclass
class ResultadoBackend:
    backend: str
    ciclos: list[ResultadoCiclo] = field(default_factory=list)

    @property
    def precision(self) -> float:
        if not self.ciclos:
            return 0.0
        return sum(1 for c in self.ciclos if c.correcto) / len(self.ciclos)

    @property
    def t_whisper_avg(self) -> float:
        return sum(c.t_whisper for c in self.ciclos) / len(self.ciclos) if self.ciclos else 0.0

    @property
    def t_llm_avg(self) -> float:
        return sum(c.t_llm for c in self.ciclos) / len(self.ciclos) if self.ciclos else 0.0

    @property
    def t_total_avg(self) -> float:
        return sum(c.t_total for c in self.ciclos) / len(self.ciclos) if self.ciclos else 0.0


def correr_ciclo(
    cmd: dict,
    audio_cap: AudioCapture,
    transcriber: Transcriber,
    llm,
    parser: ActionParser,
    context_builder: ContextBuilder,
    backend: str,
) -> ResultadoCiclo:

    resultado = ResultadoCiclo(
        backend=backend,
        comando_esperado=cmd["esperado"],
    )

    t_inicio = time.monotonic()

    # Audio
    t0 = time.monotonic()
    audio = audio_cap.record_until_silence()
    resultado.t_audio = time.monotonic() - t0

    if audio is None:
        console.print("[yellow]  ⚠ No se capturo audio[/yellow]")
        return resultado

    # Whisper
    t0 = time.monotonic()
    transcript = transcriber.transcribe(audio)
    resultado.t_whisper = time.monotonic() - t0

    if transcript is None or not transcript.text:
        console.print("[yellow]  ⚠ Transcripcion vacia[/yellow]")
        return resultado

    resultado.texto_transcrito = transcript.text
    console.print(f"  🎤 [green]'{transcript.text}'[/green]")

    # LLM
    ctx = context_builder.build(transcript.text)
    t0 = time.monotonic()
    raw = llm.generate(user_message=transcript.text, scene_context=ctx)
    resultado.t_llm = time.monotonic() - t0

    if not raw:
        console.print("[yellow]  ⚠ LLM no respondio[/yellow]")
        return resultado

    # Parser
    command, error = parser.parse_safe(raw)

    if command and command.actions:
        resultado.accion_detectada = command.actions[0].action.value
        resultado.correcto = resultado.accion_detectada == cmd["esperado"]
        estado = "[green]OK[/green]" if resultado.correcto else "[red]MAL[/red]"
        console.print(
            f"  📋 {resultado.accion_detectada} [{estado}]  "
            f"[dim]whisper={resultado.t_whisper:.1f}s  llm={resultado.t_llm:.1f}s[/dim]"
        )
    else:
        console.print(f"  [red]Parser fallo: {error}[/red]")

    resultado.t_total = time.monotonic() - t_inicio
    return resultado


def imprimir_reporte(ollama_r: ResultadoBackend, llama_r: ResultadoBackend) -> None:
    console.print("\n")
    console.rule("[bold cyan]REPORTE FINAL — Ollama vs llama.cpp[/bold cyan]")

    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Backend",    style="bold", width=12)
    table.add_column("Precisión",  justify="right", width=10)
    table.add_column("T-Whisper",  justify="right", width=10)
    table.add_column("T-LLM",      justify="right", width=10)
    table.add_column("T-Total",    justify="right", width=10)
    table.add_column("Speedup",    justify="right", width=10)

    speedup_llm = ollama_r.t_llm_avg / llama_r.t_llm_avg if llama_r.t_llm_avg > 0 else 0
    speedup_total = ollama_r.t_total_avg / llama_r.t_total_avg if llama_r.t_total_avg > 0 else 0

    table.add_row(
        "Ollama",
        f"{ollama_r.precision*100:.0f}%",
        f"{ollama_r.t_whisper_avg:.2f}s",
        f"{ollama_r.t_llm_avg:.2f}s",
        f"{ollama_r.t_total_avg:.2f}s",
        "—",
    )
    table.add_row(
        "llama.cpp",
        f"{llama_r.precision*100:.0f}%",
        f"{llama_r.t_whisper_avg:.2f}s",
        f"{llama_r.t_llm_avg:.2f}s",
        f"{llama_r.t_total_avg:.2f}s",
        f"{speedup_total:.1f}x",
        style="bold green" if llama_r.t_total_avg < ollama_r.t_total_avg else "",
    )

    console.print(table)

    ganador = "llama.cpp" if llama_r.t_total_avg < ollama_r.t_total_avg else "Ollama"
    mejor_precision = "llama.cpp" if llama_r.precision >= ollama_r.precision else "Ollama"

    console.print(Panel.fit(
        f"[bold]Velocidad:[/bold] [cyan]{ganador}[/cyan] es más rápido "
        f"({speedup_total:.1f}x speedup en total)\n"
        f"[bold]Precisión:[/bold] [cyan]{mejor_precision}[/cyan] "
        f"(Ollama={ollama_r.precision*100:.0f}%  llama.cpp={llama_r.precision*100:.0f}%)\n"
        f"[bold]LLM speedup:[/bold] {speedup_llm:.1f}x más rápido en inferencia",
        border_style="green",
        title="Conclusión",
    ))


def guardar_json(
    ollama_r: ResultadoBackend,
    llama_r: ResultadoBackend,
    path: str = "benchmark_llama_cpp.json",
) -> None:
    datos = {
        "whisper_model": "small",
        "backends": [
            {
                "backend": r.backend,
                "precision": round(r.precision, 3),
                "t_whisper_avg": round(r.t_whisper_avg, 3),
                "t_llm_avg": round(r.t_llm_avg, 3),
                "t_total_avg": round(r.t_total_avg, 3),
                "ciclos": [
                    {
                        "esperado": c.comando_esperado,
                        "transcrito": c.texto_transcrito,
                        "detectado": c.accion_detectada,
                        "correcto": c.correcto,
                        "t_whisper": round(c.t_whisper, 3),
                        "t_llm": round(c.t_llm, 3),
                        "t_total": round(c.t_total, 3),
                    }
                    for c in r.ciclos
                ],
            }
            for r in [ollama_r, llama_r]
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Resultados guardados en: {path}[/dim]")


def main() -> None:
    cfg_path = os.environ.get("RVC_CONFIG", "config/settings.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    console.print(Panel.fit(
        "[bold cyan]Benchmark — Ollama vs llama.cpp[/bold cyan]\n"
        "[dim]Whisper small · llama3.2 · 8 comandos por backend[/dim]",
        border_style="cyan",
    ))

    input("\nPresiona ENTER para comenzar...")

    parser = ActionParser()
    context_builder = ContextBuilder()
    audio_cap = AudioCapture(cfg["audio"])

    # Whisper small para ambos
    whisper_cfg = dict(cfg["whisper"])
    whisper_cfg["model_size"] = "small"

    with Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}[/dim]"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        t = progress.add_task("Cargando Whisper small...", total=None)
        transcriber = Transcriber(whisper_cfg)
        progress.update(t, description="Listo")

    resultados = {}

    for backend_name, backend_cfg_key in [("Ollama", "ollama"), ("llama.cpp", "llama_cpp")]:
        console.rule(f"[bold yellow]Backend: {backend_name}[/bold yellow]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            t = progress.add_task(f"Cargando {backend_name}...", total=None)
            if backend_name == "Ollama":
                llm = LLMClient(cfg["ollama"], cfg["robot"])
            else:
                llm = LlamaCppClient(cfg["llama_cpp"], cfg["robot"])
            progress.update(t, description="Listo")

        resultado = ResultadoBackend(backend=backend_name)

        for i, cmd in enumerate(COMANDOS, 1):
            console.print(f"\n  [{i}/{len(COMANDOS)}] Di: [bold]'{cmd['di']}'[/bold]")
            input("  Presiona ENTER cuando estés listo...")

            ciclo = correr_ciclo(
                cmd=cmd,
                audio_cap=audio_cap,
                transcriber=transcriber,
                llm=llm,
                parser=parser,
                context_builder=context_builder,
                backend=backend_name,
            )
            resultado.ciclos.append(ciclo)

        resultados[backend_name] = resultado
        console.print(
            f"\n  [green]Precisión {backend_name}: "
            f"{resultado.precision*100:.0f}%[/green]  "
            f"[dim]T-LLM avg: {resultado.t_llm_avg:.2f}s[/dim]"
        )

    imprimir_reporte(resultados["Ollama"], resultados["llama.cpp"])
    guardar_json(resultados["Ollama"], resultados["llama.cpp"])


if __name__ == "__main__":
    main()