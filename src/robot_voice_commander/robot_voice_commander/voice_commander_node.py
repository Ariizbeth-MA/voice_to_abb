"""ROS2 node — runs the voice pipeline and publishes RobotCommand messages."""

from __future__ import annotations

import json
import rclpy
from rclpy.node import Node
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from robot_voice_msgs.msg import RobotCommand, RobotAction, PipelineStatus

from .pipeline import VoiceCommandPipeline
from .modules.audio import AudioCapture, Transcriber
from .modules.llm import LLMClient
from .modules.parser import ActionParser
from .modules.context import ContextBuilder
from .modules.llm import LLMClient, LlamaCppClient

import yaml
import os

console = Console()


class VoiceCommanderNode(Node):

    def __init__(self):
        super().__init__('voice_commander')

        # Publishers
        self.cmd_pub = self.create_publisher(
            RobotCommand, '/robot/voice_commands', 10)
        self.status_pub = self.create_publisher(
            PipelineStatus, '/robot/pipeline_status', 10)

        # Config
        cfg_path = os.environ.get('RVC_CONFIG', 'config/settings.yaml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Autodetectar hardware
        from .modules.hardware import get_hardware_config, detect_cuda
        cfg = get_hardware_config(cfg)
        cuda = cfg['hardware']['cuda']
        device_str = "[green]GPU NVIDIA (CUDA)[/green]" if cuda else "[yellow]CPU[/yellow]"
        console.print(f"[bold]Hardware:[/bold] {device_str}")

        console.print(Panel.fit(
            "[bold cyan]Robot Voice Commander[/bold cyan]\n"
            "[dim]ROS2 Jazzy · Whisper STT · Ollama LLM[/dim]",
            border_style="cyan",
        ))

        console.print("[dim]Cargando modulos...[/dim]")

        # Pipeline
        self.pipeline = VoiceCommandPipeline(
            audio_capture=AudioCapture(cfg['audio']),
            transcriber=Transcriber(cfg['whisper']),
            llm_client=LlamaCppClient(cfg['llama_cpp'], cfg['robot']),
            action_parser=ActionParser(),
            context_builder=ContextBuilder(),
        )

        console.print("[green]Todos los modulos listos.[/green]\n")

        self.timer = self.create_timer(0.1, self.run_cycle)

    def run_cycle(self):
        self.timer.cancel()

        console.rule("[bold yellow]Esperando comando de voz...[/bold yellow]")
        console.print("[dim]Habla cuando quieras[/dim]\n")

        ctx = self.pipeline.run_cycle()

        # Siempre muestra la transcripcion
        if ctx.transcription:
            console.print(
                f"[bold]🎤 Escuché:[/bold] [green italic]'{ctx.transcription}'[/green italic]"
            )
        else:
            console.print("[yellow]No se detectó voz, intenta de nuevo.[/yellow]")
            self.timer.reset()
            return

        # Muestra respuesta del LLM
        if ctx.llm_raw_response:
            console.print(f"[bold]🤖 LLM respondió:[/bold] [dim]{ctx.llm_raw_response[:120]}...[/dim]")

        # Error de parse
        if ctx.parse_error:
            console.print(f"[red bold]❌ Error:[/red bold] {ctx.parse_error}")
            self.timer.reset()
            return

        # Comando parseado exitoso
        if ctx.success:
            cmd = ctx.parsed_command
            conf_color = "green" if cmd.confidence >= 0.7 else "yellow"

            console.print(
                f"[bold]📋 Intent:[/bold] {cmd.intent}  "
                f"[{conf_color}]confianza={cmd.confidence:.2f}[/{conf_color}]"
            )

            if cmd.clarification_needed:
                console.print(
                    f"[yellow bold]❓ Necesito aclaración:[/yellow bold] {cmd.clarification_message}"
                )
            else:
                table = Table(
                    show_header=True,
                    header_style="bold magenta",
                    box=None,
                    padding=(0, 2),
                )
                table.add_column("#", style="dim", width=3)
                table.add_column("Acción", style="bold")
                table.add_column("Parámetros")

                for i, action in enumerate(cmd.actions, 1):
                    params_str = ", ".join(f"{k}={v}" for k, v in action.parameters.items())
                    table.add_row(str(i), action.action.value, params_str)

                console.print(table)

            # Publicar en ROS2
            self._publish_command(ctx)
            console.print("[green]✓ Publicado en /robot/voice_commands[/green]\n")

        self.timer.reset()

    def _publish_command(self, ctx):
        cmd = ctx.parsed_command

        # Status
        status = PipelineStatus()
        status.cycle_id = ctx.cycle_id
        status.success = ctx.success
        status.transcription = ctx.transcription or ''
        status.error_message = ctx.parse_error or ''
        status.audio_duration = float(ctx.raw_audio_duration)
        self.status_pub.publish(status)

        # Comando
        msg = RobotCommand()
        msg.intent = cmd.intent
        msg.confidence = float(cmd.confidence)
        msg.clarification_needed = cmd.clarification_needed
        msg.clarification_message = cmd.clarification_message or ''

        for a in cmd.actions:
            action_msg = RobotAction()
            action_msg.action = a.action.value
            action_msg.parameters_json = json.dumps(a.parameters)
            msg.actions.append(action_msg)

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline detenido.[/yellow]")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()