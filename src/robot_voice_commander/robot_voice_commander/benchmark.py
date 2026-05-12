"""
Benchmark de combinaciones Whisper + Ollama.
Mide latencia y precision para elegir la mejor combinacion.
"""

from __future__ import annotations

import json
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .modules.audio import AudioCapture, Transcriber
from .modules.llm import LLMClient
from .modules.parser import ActionParser
from .modules.context import ContextBuilder

console = Console()

# Modelos comparados 
WHISPER_MODELS = ["tiny", "base", "small"]
OLLAMA_MODELS  = ["llama3.2", "mistral", "phi3"]

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
    whisper_model: str
    ollama_model: str
    comando_esperado: str
    texto_transcrito: str = ""
    accion_detectada: str = ""
    correcto: bool = False
    t_audio: float = 0.0
    t_whisper: float = 0.0
    t_llm: float = 0.0
    t_parser: float = 0.0
    t_total: float = 0.0


@dataclass
class ResultadoCombinacion:
    whisper_model: str
    ollama_model: str
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

    @property
    def score(self) -> float:
        """Score combinado: 60% precision, 40% velocidad (normalizada)."""
        velocidad = 1.0 / (self.t_total_avg + 0.001)
        return 0.6 * self.precision + 0.4 * min(velocidad, 1.0)


def cargar_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def correr_ciclo(
    cmd: dict,
    audio_cap: AudioCapture,
    transcriber: Transcriber,
    llm: LLMClient,
    parser: ActionParser,
    context_builder: ContextBuilder,
    whisper_model: str,
    ollama_model: str,
) -> ResultadoCiclo:

    resultado = ResultadoCiclo(
        whisper_model=whisper_model,
        ollama_model=ollama_model,
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
    t0 = time.monotonic()
    command, error = parser.parse_safe(raw)
    resultado.t_parser = time.monotonic() - t0

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


def imprimir_reporte(resultados: list[ResultadoCombinacion]) -> None:
    console.print("\n")
    console.rule("[bold cyan]REPORTE FINAL[/bold cyan]")

    # Tabla resumen
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Whisper",    style="bold", width=8)
    table.add_column("Ollama",     style="bold", width=12)
    table.add_column("Precisión",  justify="right", width=10)
    table.add_column("T-Whisper",  justify="right", width=10)
    table.add_column("T-LLM",      justify="right", width=8)
    table.add_column("T-Total",    justify="right", width=8)
    table.add_column("Score",      justify="right", width=8)

    ordenados = sorted(resultados, key=lambda r: r.score, reverse=True)

    for i, r in enumerate(ordenados):
        estilo = "bold green" if i == 0 else ""
        prefix = "★ " if i == 0 else "  "
        table.add_row(
            prefix + r.whisper_model,
            r.ollama_model,
            f"{r.precision*100:.0f}%",
            f"{r.t_whisper_avg:.2f}s",
            f"{r.t_llm_avg:.2f}s",
            f"{r.t_total_avg:.2f}s",
            f"{r.score:.3f}",
            style=estilo,
        )

    console.print(table)

    mejor = ordenados[0]
    console.print(Panel.fit(
        f"[bold green]Mejor combinación:[/bold green]\n"
        f"  Whisper : [cyan]{mejor.whisper_model}[/cyan]\n"
        f"  Ollama  : [cyan]{mejor.ollama_model}[/cyan]\n"
        f"  Precisión: {mejor.precision*100:.0f}%  |  "
        f"Total promedio: {mejor.t_total_avg:.2f}s",
        border_style="green",
    ))


def guardar_json(resultados: list[ResultadoCombinacion], path: str = "benchmark_resultados.json") -> None:
    datos = []
    for r in resultados:
        datos.append({
            "whisper": r.whisper_model,
            "ollama": r.ollama_model,
            "precision": round(r.precision, 3),
            "t_whisper_avg": round(r.t_whisper_avg, 3),
            "t_llm_avg": round(r.t_llm_avg, 3),
            "t_total_avg": round(r.t_total_avg, 3),
            "score": round(r.score, 4),
            "ciclos": [
                {
                    "esperado": c.comando_esperado,
                    "transcrito": c.texto_transcrito,
                    "detectado": c.accion_detectada,
                    "correcto": c.correcto,
                    "t_audio": round(c.t_audio, 3),
                    "t_whisper": round(c.t_whisper, 3),
                    "t_llm": round(c.t_llm, 3),
                    "t_total": round(c.t_total, 3),
                }
                for c in r.ciclos
            ],
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Resultados guardados en: {path}[/dim]")


def main() -> None:
    import os
    cfg_path = os.environ.get("RVC_CONFIG", "config/settings.yaml")
    cfg = cargar_config(cfg_path)

    console.print(Panel.fit(
        "[bold cyan]Benchmark — Whisper × Ollama[/bold cyan]\n"
        f"[dim]{len(WHISPER_MODELS)} modelos Whisper × "
        f"{len(OLLAMA_MODELS)} modelos Ollama × "
        f"{len(COMANDOS)} comandos[/dim]",
        border_style="cyan",
    ))

    total_combinaciones = len(WHISPER_MODELS) * len(OLLAMA_MODELS)
    console.print(f"\n[bold]Total combinaciones:[/bold] {total_combinaciones}")
    console.print(f"[bold]Comandos por combinación:[/bold] {len(COMANDOS)}")
    console.print(f"[bold]Ciclos totales:[/bold] {total_combinaciones * len(COMANDOS)}\n")

    input("Presiona ENTER para comenzar...")

    parser = ActionParser()
    context_builder = ContextBuilder()
    audio_cap = AudioCapture(cfg["audio"])

    resultados: list[ResultadoCombinacion] = []
    combinacion_num = 0

    for whisper_model in WHISPER_MODELS:
        for ollama_model in OLLAMA_MODELS:
            combinacion_num += 1

            console.rule(
                f"[bold yellow]Combinación {combinacion_num}/{total_combinaciones}: "
                f"whisper={whisper_model}  ollama={ollama_model}[/bold yellow]"
            )

            # Cargar modelos para esta combinacion
            whisper_cfg = dict(cfg["whisper"])
            whisper_cfg["model_size"] = whisper_model

            ollama_cfg = dict(cfg["ollama"])
            ollama_cfg["model"] = ollama_model

            with Progress(
                SpinnerColumn(),
                TextColumn("[dim]{task.description}[/dim]"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                t = progress.add_task("Cargando Whisper...", total=None)
                transcriber = Transcriber(whisper_cfg)
                progress.update(t, description="Cargando Ollama...")
                llm = LLMClient(ollama_cfg, cfg["robot"])
                progress.update(t, description="Listo")

            resultado_combinacion = ResultadoCombinacion(
                whisper_model=whisper_model,
                ollama_model=ollama_model,
            )

            for i, cmd in enumerate(COMANDOS, 1):
                console.print(
                    f"\n  [{i}/{len(COMANDOS)}] Di: [bold]'{cmd['di']}'[/bold]"
                )
                input("  Presiona ENTER cuando estés listo...")

                ciclo = correr_ciclo(
                    cmd=cmd,
                    audio_cap=audio_cap,
                    transcriber=transcriber,
                    llm=llm,
                    parser=parser,
                    context_builder=context_builder,
                    whisper_model=whisper_model,
                    ollama_model=ollama_model,
                )
                resultado_combinacion.ciclos.append(ciclo)

            resultados.append(resultado_combinacion)
            console.print(
                f"\n  [green]Precisión esta combinación: "
                f"{resultado_combinacion.precision*100:.0f}%[/green]  "
                f"[dim]T-total avg: {resultado_combinacion.t_total_avg:.2f}s[/dim]"
            )

    imprimir_reporte(resultados)
    guardar_json(resultados)


if __name__ == "__main__":
    main()