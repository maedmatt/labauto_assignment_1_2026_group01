
import os
import sys
import time
import yaml

from scipy.io import savemat
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if sys.platform == "darwin" and os.environ.get("MJPYTHON_BIN"):
    # Conda's mjpython wrapper keeps sys.executable as python; labauto checks for mjpython.
    sys.executable = os.environ["MJPYTHON_BIN"]

from labauto import MuJoCoMechanicalSystem
from labauto import TrapezoidalMotionLaw
from labauto import loadController
from labauto import loadInstructions
import random
from scipy.signal import chirp

model_name = "gantry_portal_sea_soft"


# Load controller parameters and dynamic parameters
with open(f'{model_name}/initial_control_config.yaml', 'r') as file:
    params_yaml = yaml.safe_load(file)
    controller_params = params_yaml['controller']
    dynamic_params = np.array(params_yaml['model_parameters'])
    motion_law_params = controller_params['motion_law_parameters']

# Create simulator for Gantry SEA robot (MuJoCo)
xml_path = f"{model_name}/model_without_vases.xml"
robot = MuJoCoMechanicalSystem(xml_path=xml_path)
robot.initialize()
robot.show()
dof = robot.get_input_number()

# Set the cycle time (sampling time) for motion law updates
Tc = robot.get_sampling_period()

# define chirp
Duration = 20.0 # seconds
t = np.arange(0, Duration + Tc, Tc)  # Ensure inclusion of Duration if possible


# Load the tuned controller using parameters from YAML
decentralized_ctrl=loadController(Tc,controller_params,dynamic_params,model_name)
decentralized_ctrl.initialize()
decentralized_ctrl.set_umax(robot.get_umax())

# Initial reference is equal to the initial state of the robot
measured_output = robot.read_sensor_value()
q0 = measured_output[:dof]
Dq0 = measured_output[dof:]
DDq0 = np.array([0]*dof)
initial_reference = np.concatenate((q0, Dq0, DDq0))
# Read the initial torque
joint_torque = robot.read_actuator_value()
feedforward_action = np.array([0.0]*dof)
decentralized_ctrl.starting(initial_reference, measured_output, joint_torque, feedforward_action)

for itest in range(0,40):


    ml = TrapezoidalMotionLaw(motion_law_params, Tc)
    measured_output = robot.read_sensor_value()
    q0 = measured_output[:3]
    ml.set_initial_condition(q0)

    # Define a sequence of motion instructions
    initial_position = [0.0] * dof
    instructions = ["pause: 1", f"move: {initial_position}", "pause: 5"]
    ml.add_instructions(instructions)

    f0 = 0.1+9.9*np.random.rand(1)[0] # random number between 0.1 and 10
    f1 = 100.0+400.0*np.random.rand(1)[0] # random number between 100 and 500
    A = 50.0+50.0*np.random.rand(1)[0] # random number between 5 and 15
    joint_number = np.random.randint(dof) # if random number <0.5, use joint 0,  otherwise joint 1

    if np.random.rand(1)[0]>0.5: # 50% chance to invert f0 and f1
        tmp=f0
        f0=f1
        f1=tmp

    options = ['linear', 'quadratic', 'logarithmic', 'hyperbolic']
    # Pick a random choice
    random_choice = random.choice(options)

    print(f"Running experiment {itest}, joint {joint_number}, f0 ={f0}, f1={f1}")

    chirp_signal = A * chirp(t, f0=f0, f1=f1, t1=Duration, method=random_choice)

    # Simulation loop
    while ml.depending_instructions():
        loop_t0 = time.perf_counter()

        target_q, target_Dq, target_DDq = ml.compute_motion_law()
        reference = np.concatenate((target_q, target_Dq, target_DDq))
        measured_output = robot.read_sensor_value()
        joint_torque = decentralized_ctrl.compute_control_action(reference, measured_output, feedforward_action)
        robot.write_actuator_value(joint_torque)
        robot.simulate()

        computation_time = time.perf_counter() - loop_t0
        time.sleep(max(0.0, Tc - computation_time))


    measured_signal, control_action=  [], []
    feedforward_action = np.array([0.0]*dof)

    for actual_time,disturbance in zip(t,chirp_signal):
        loop_t0 = time.perf_counter()
        target_q, target_Dq, target_DDq = ml.compute_motion_law()
        reference = np.concatenate((target_q, target_Dq, target_DDq))
        measured_output = robot.read_sensor_value()
        feedforward_action[joint_number]= disturbance
        joint_torque = decentralized_ctrl.compute_control_action(reference, measured_output, feedforward_action)
        robot.write_actuator_value(joint_torque)

        # Store data
        measured_signal.append(measured_output)
        control_action.append(joint_torque)

        robot.simulate()
        computation_time = time.perf_counter() - loop_t0
        time.sleep(max(0.0, Tc - computation_time))

    # Convert lists to numpy arrays
    measured_signal = np.array(measured_signal)
    control_action = np.array(control_action)

    # Post-processing
    joint_position = measured_signal[:, :dof]
    joint_velocity = measured_signal[:, dof:]

    # Save test data
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    test_data = {
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "joint_torque": control_action,
        "chirp_signal": chirp_signal,
        "joint_number": joint_number,
        "f0": f0,
        "f1": f1,
        "A": A,
        "time": t
    }
    savemat(f"{model_name}/tests/validation_chirp_experiment_joint{joint_number+1}_{timestamp}.mat", {key: test_data[key] for key in test_data})

    from labauto import plot_frf

    fig_frf, frf = plot_frf(
        joint_velocity[:, joint_number].flatten(), control_action[:, joint_number].flatten(), 1.0 / Tc,
        nperseg=1024,
        max_freq=2000
    )

    fig_frf.show()

robot.close()
