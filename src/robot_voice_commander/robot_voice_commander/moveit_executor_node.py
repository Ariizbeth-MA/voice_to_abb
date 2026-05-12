"""ROS2 node — recibe RobotCommand y ejecuta movimientos con MoveIt2."""

from __future__ import annotations

import json
import rclpy
from rclpy.node import Node
from robot_voice_msgs.msg import RobotCommand

from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


class MoveItExecutorNode(Node):

    def __init__(self):
        super().__init__('moveit_executor')

        # MoveIt2
        self.moveit = MoveItPy(node_name='moveit_executor')
        self.arm = self.moveit.get_planning_component('manipulator')
        self.get_logger().info('MoveIt2 listo.')

        # Suscriptor
        self.sub = self.create_subscription(
            RobotCommand,
            '/robot/voice_commands',
            self.on_command,
            10,
        )
        self.get_logger().info('MoveItExecutor esperando comandos...')

    def on_command(self, msg: RobotCommand):
        self.get_logger().info(f'Comando recibido: {msg.intent}')

        if msg.clarification_needed:
            self.get_logger().warn(
                f'Clarificacion necesaria: {msg.clarification_message}')
            return

        if msg.confidence < 0.5:
            self.get_logger().warn(
                f'Confianza baja ({msg.confidence:.2f}), ignorando.')
            return

        for action in msg.actions:
            self.execute_action(action.action, json.loads(action.parameters_json))

    def execute_action(self, action: str, params: dict):
        self.get_logger().info(f'Ejecutando: {action} — {params}')

        if action == 'move_home':
            self._move_to_named('home')

        elif action == 'move_joint':
            self._move_joint(params)

        elif action == 'move_cartesian':
            self._move_cartesian(params)

        elif action == 'open_gripper':
            self._move_to_named('open')

        elif action == 'close_gripper':
            self._move_to_named('close')

        elif action == 'stop':
            self.get_logger().warn('STOP recibido — deteniendo.')
            self.arm.stop()

        else:
            self.get_logger().warn(f'Accion no implementada: {action}')

    def _move_to_named(self, target: str):
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name=target)
        self._plan_and_execute()

    def _move_joint(self, params: dict):
        """Mueve un joint a un angulo absoluto."""
        robot_state = self.moveit.get_robot_state()
        joint = params.get('joint', '')
        angle_deg = params.get('angle', 0.0)
        angle_rad = angle_deg * 3.14159 / 180.0

        robot_state.set_joint_positions({joint: angle_rad})
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=robot_state)
        self._plan_and_execute()

    def _move_cartesian(self, params: dict):
        """Ejecuta un movimiento cartesiano relativo."""
        from geometry_msgs.msg import Pose
        pose = Pose()

        current = self.moveit.get_robot_state()
        # Por ahora solo traslacion en X, Y, Z
        pose.position.x = float(params.get('x', 0.0))
        pose.position.y = float(params.get('y', 0.0))
        pose.position.z = float(params.get('z', 0.0))
        pose.orientation.w = 1.0

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pose, pose_link='tool0')
        self._plan_and_execute()

    def _plan_and_execute(self):
        plan = self.arm.plan()
        if plan:
            self.moveit.execute(plan.trajectory, controllers=[])
            self.get_logger().info('Movimiento ejecutado.')
        else:
            self.get_logger().error('Fallo la planeacion.')


def main(args=None):
    rclpy.init(args=args)
    node = MoveItExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()