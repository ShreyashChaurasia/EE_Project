from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Simple in-memory cache for plots
plot_cache = {}

def generate_plot(R, L, C, frequency, voltage, plot_type='Vt'):
    try:
        # Create a cache key based on input parameters and plot type.
        cache_key = f"{R}_{L}_{C}_{frequency}_{voltage}_{plot_type}"
        if cache_key in plot_cache:
            return plot_cache[cache_key]

        # Validate parameters
        if R <= 0 or frequency <= 0 or voltage <= 0:
            raise ValueError("R, frequency, and voltage must be positive values")
        if L < 0:
            raise ValueError("Inductance cannot be negative")
        if C <= 0:
            raise ValueError("Capacitance must be positive")

        omega = 2 * np.pi * frequency
        t = np.linspace(0, 0.1, 500)

        XL = omega * L
        XC = 1 / (omega * C)
        Z = np.sqrt(R**2 + (XL - XC)**2)
        phi = np.arctan2((XL - XC), R)

        # Balanced three-phase voltages and currents
        Vr = voltage * np.sin(omega * t)
        Vy = voltage * np.sin(omega * t - 2 * np.pi / 3)
        Vb = voltage * np.sin(omega * t - 4 * np.pi / 3)

        Ir = (voltage / Z) * np.sin(omega * t - phi)
        Iy = (voltage / Z) * np.sin(omega * t - 2 * np.pi / 3 - phi)
        Ib = (voltage / Z) * np.sin(omega * t - 4 * np.pi / 3 - phi)

        # Compute power waveforms
        Pr = Vr * Ir      # Active Power
        Qr = Vr * Ir * np.sin(phi)  # Reactive Power
        Sr = Vr * Ir      # Apparent Power

        Py = Vy * Iy
        Qy = Vy * Iy * np.sin(phi)
        Sy = Vy * Iy

        Pb = Vb * Ib
        Qb = Vb * Ib * np.sin(phi)
        Sb = Vb * Ib

        fig, ax = plt.subplots(figsize=(10, 8))
        if plot_type == 'Vt':
            ax.plot(t, Vr, 'r', label='Phase R')
            ax.plot(t, Vy, 'y', label='Phase Y')
            ax.plot(t, Vb, 'b', label='Phase B')
            ax.set_title(f'Three-Phase Voltages (Z = {Z:.2f} Ω)')
            ax.set_ylabel('Voltage (V)')
        elif plot_type == 'It':
            ax.plot(t, Ir, 'r', label='Current R')
            ax.plot(t, Iy, 'y', label='Current Y')
            ax.plot(t, Ib, 'b', label='Current B')
            ax.set_title(f'Three-Phase Currents (φ = {np.degrees(phi):.2f}°)')
            ax.set_ylabel('Current (A)')
        elif plot_type == 'Pt':
            ax.plot(t, Pr, 'r', label='Active Power R')
            ax.plot(t, Py, 'y', label='Active Power Y')
            ax.plot(t, Pb, 'b', label='Active Power B')
            ax.set_title('Three-Phase Active Power')
            ax.set_ylabel('Power (W)')
        elif plot_type == 'Qt':
            ax.plot(t, Qr, 'r', label='Reactive Power R')
            ax.plot(t, Qy, 'y', label='Reactive Power Y')
            ax.plot(t, Qb, 'b', label='Reactive Power B')
            ax.set_title('Three-Phase Reactive Power')
            ax.set_ylabel('Power (VAR)')
        elif plot_type == 'St':
            ax.plot(t, Sr, 'r', label='Apparent Power R')
            ax.plot(t, Sy, 'y', label='Apparent Power Y')
            ax.plot(t, Sb, 'b', label='Apparent Power B')
            ax.set_title('Three-Phase Apparent Power')
            ax.set_ylabel('Power (VA)')
        else:
            raise ValueError("Invalid plot type")
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
        plt.close()
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plot_cache[cache_key] = plot_data
        return plot_data
    except Exception as e:
        app.logger.error(f"Error in generate_plot: {e}")
        raise

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/update_all_plots', methods=['POST'])
def update_all_plots():
    try:
        data = request.get_json()
        app.logger.debug(f"Received data for all plots: {data}")
        required = ['R', 'L', 'C', 'frequency', 'voltage']
        if not all(key in data for key in required):
            raise ValueError("Missing required parameters")
        R = float(data['R'])
        L = float(data['L'])
        C = float(data['C'])
        frequency = float(data['frequency'])
        voltage = float(data['voltage'])
        plotTypes = ['Vt', 'It', 'Pt', 'Qt', 'St']
        plots = {}
        for pt in plotTypes:
            plots[pt] = generate_plot(R, L, C, frequency, voltage, pt)
        return jsonify(plots)
    except Exception as e:
        app.logger.error(f"Error in update_all_plots: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/calculations', methods=['POST'])
def calculations():
    try:
        data = request.get_json()
        app.logger.debug(f"Calculations data: {data}")
        required = ['R', 'L', 'C', 'frequency', 'voltage']
        if not all(key in data for key in required):
            raise ValueError("Missing required parameters")
        R = float(data['R'])
        L = float(data['L'])
        C = float(data['C'])
        frequency = float(data['frequency'])
        voltage = float(data['voltage'])
        omega = 2 * np.pi * frequency
        XL = omega * L
        XC = 1 / (omega * C)
        Z = np.sqrt(R**2 + (XL - XC)**2)
        phi = np.arctan2((XL - XC), R)
        calc_details = {
            "Angular Frequency (ω)": omega,
            "Inductive Reactance (X_L)": XL,
            "Capacitive Reactance (X_C)": XC,
            "Impedance (Z)": Z,
            "Phase Angle (φ in radians)": phi,
            "Phase Angle (φ in degrees)": np.degrees(phi)
        }
        if 'time' in data:
            t_val = float(data['time'])
            Vr = voltage * np.sin(omega * t_val)
            Vy = voltage * np.sin(omega * t_val - 2 * np.pi / 3)
            Vb = voltage * np.sin(omega * t_val - 4 * np.pi / 3)
            Ir = (voltage / Z) * np.sin(omega * t_val - phi)
            Iy = (voltage / Z) * np.sin(omega * t_val - 2 * np.pi / 3 - phi)
            Ib = (voltage / Z) * np.sin(omega * t_val - 4 * np.pi / 3 - phi)
            Pr = Vr * Ir
            Py = Vy * Iy
            Pb = Vb * Ib
            calc_details.update({
                "Instantaneous Time (s)": t_val,
                "Voltage R (V)": Vr,
                "Voltage Y (V)": Vy,
                "Voltage B (V)": Vb,
                "Current R (A)": Ir,
                "Current Y (A)": Iy,
                "Current B (A)": Ib,
                "Active Power R (W)": Pr,
                "Active Power Y (W)": Py,
                "Active Power B (W)": Pb
            })
        return jsonify(calc_details)
    except Exception as e:
        app.logger.error(f"Error in calculations endpoint: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/update_unbalanced_plots', methods=['POST'])
def update_unbalanced_plots():
    try:
        data = request.get_json()
        app.logger.debug(f"Received unbalanced data: {data}")
        # Required parameters: Separate R, L, C for each phase; frequency and voltage are common.
        required = ['R_R', 'R_Y', 'R_B', 'L_R', 'L_Y', 'L_B', 'C_R', 'C_Y', 'C_B', 'frequency', 'voltage']
        if not all(key in data for key in required):
            raise ValueError("Missing required parameters for unbalanced load")
        # Convert inputs for phase R
        R_R = float(data['R_R'])
        L_R = float(data['L_R'])
        C_R = float(data['C_R'])
        # Convert inputs for phase Y
        R_Y = float(data['R_Y'])
        L_Y = float(data['L_Y'])
        C_Y = float(data['C_Y'])
        # Convert inputs for phase B
        R_B = float(data['R_B'])
        L_B = float(data['L_B'])
        C_B = float(data['C_B'])
        frequency = float(data['frequency'])
        voltage = float(data['voltage'])
        omega = 2 * np.pi * frequency

        # For each phase, calculate impedance and phase angle using the phase‑specific values.
        def calc_phase(R, L, C):
            XL = omega * L
            XC = 1 / (omega * C)
            Z = np.sqrt(R**2 + (XL - XC)**2)
            phi = np.arctan2((XL - XC), R)
            return Z, phi

        Z_R, phi_R = calc_phase(R_R, L_R, C_R)
        Z_Y, phi_Y = calc_phase(R_Y, L_Y, C_Y)
        Z_B, phi_B = calc_phase(R_B, L_B, C_B)

        t = np.linspace(0, 0.1, 500)

        # Use common voltage waveforms (with standard phase shifts)
        Vr = voltage * np.sin(omega * t)
        Vy = voltage * np.sin(omega * t - 2 * np.pi / 3)
        Vb = voltage * np.sin(omega * t - 4 * np.pi / 3)

        # Compute currents for each phase with its own impedance and phase angle.
        Ir = (voltage / Z_R) * np.sin(omega * t - phi_R)
        Iy = (voltage / Z_Y) * np.sin(omega * t - 2 * np.pi / 3 - phi_Y)
        Ib = (voltage / Z_B) * np.sin(omega * t - 4 * np.pi / 3 - phi_B)

        plotTypes = ['Vt', 'It', 'Pt', 'Qt', 'St']
        plots = {}
        for pt in plotTypes:
            fig, ax = plt.subplots(figsize=(10, 8))
            if pt == 'Vt':
                ax.plot(t, Vr, 'r', label='Phase R')
                ax.plot(t, Vy, 'y', label='Phase Y')
                ax.plot(t, Vb, 'b', label='Phase B')
                ax.set_title('Unbalanced Three-Phase Voltages')
                ax.set_ylabel('Voltage (V)')
            elif pt == 'It':
                ax.plot(t, Ir, 'r', label='Current R')
                ax.plot(t, Iy, 'y', label='Current Y')
                ax.plot(t, Ib, 'b', label='Current B')
                ax.set_title('Unbalanced Three-Phase Currents')
                ax.set_ylabel('Current (A)')
            elif pt == 'Pt':
                ax.plot(t, Vr*Ir, 'r', label='Active Power R')
                ax.plot(t, Vy*Iy, 'y', label='Active Power Y')
                ax.plot(t, Vb*Ib, 'b', label='Active Power B')
                ax.set_title('Unbalanced Three-Phase Active Power')
                ax.set_ylabel('Power (W)')
            elif pt == 'Qt':
                ax.plot(t, Vr*Ir*np.sin(phi_R), 'r', label='Reactive Power R')
                ax.plot(t, Vy*Iy*np.sin(phi_Y), 'y', label='Reactive Power Y')
                ax.plot(t, Vb*Ib*np.sin(phi_B), 'b', label='Reactive Power B')
                ax.set_title('Unbalanced Three-Phase Reactive Power')
                ax.set_ylabel('Power (VAR)')
            elif pt == 'St':
                ax.plot(t, Vr*Ir, 'r', label='Apparent Power R')
                ax.plot(t, Vy*Iy, 'y', label='Apparent Power Y')
                ax.plot(t, Vb*Ib, 'b', label='Apparent Power B')
                ax.set_title('Unbalanced Three-Phase Apparent Power')
                ax.set_ylabel('Power (VA)')
            else:
                raise ValueError("Invalid plot type")
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid(True)
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
            plt.close()
            plots[pt] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify(plots)
    except Exception as e:
        app.logger.error(f"Error in update_unbalanced_plots: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/calculations_unbalanced', methods=['POST'])
def calculations_unbalanced():
    try:
        data = request.get_json()
        app.logger.debug(f"Unbalanced Calculations data: {data}")
        required = ['R_R', 'R_Y', 'R_B', 'L_R', 'L_Y', 'L_B', 'C_R', 'C_Y', 'C_B', 'frequency', 'voltage']
        if not all(key in data for key in required):
            raise ValueError("Missing required parameters for unbalanced load")
        # Convert inputs for phase R
        R_R = float(data['R_R'])
        L_R = float(data['L_R'])
        C_R = float(data['C_R'])
        # For phase Y
        R_Y = float(data['R_Y'])
        L_Y = float(data['L_Y'])
        C_Y = float(data['C_Y'])
        # For phase B
        R_B = float(data['R_B'])
        L_B = float(data['L_B'])
        C_B = float(data['C_B'])
        frequency = float(data['frequency'])
        voltage = float(data['voltage'])
        omega = 2 * np.pi * frequency

        def calc_phase(R, L, C):
            XL = omega * L
            XC = 1 / (omega * C)
            Z = np.sqrt(R**2 + (XL - XC)**2)
            phi = np.arctan2((XL - XC), R)
            return Z, phi

        Z_R, phi_R = calc_phase(R_R, L_R, C_R)
        Z_Y, phi_Y = calc_phase(R_Y, L_Y, C_Y)
        Z_B, phi_B = calc_phase(R_B, L_B, C_B)

        calc_details = {
            "Angular Frequency (ω)": omega,
            "Phase R Impedance (Z_R)": Z_R,
            "Phase Y Impedance (Z_Y)": Z_Y,
            "Phase B Impedance (Z_B)": Z_B,
            "Phase R Angle (φ_R in degrees)": np.degrees(phi_R),
            "Phase Y Angle (φ_Y in degrees)": np.degrees(phi_Y),
            "Phase B Angle (φ_B in degrees)": np.degrees(phi_B)
        }

        if 'time' in data:
            t_val = float(data['time'])
            Vr = voltage * np.sin(omega * t_val)
            Vy = voltage * np.sin(omega * t_val - 2 * np.pi / 3)
            Vb = voltage * np.sin(omega * t_val - 4 * np.pi / 3)
            Ir = (voltage / Z_R) * np.sin(omega * t_val - phi_R)
            Iy = (voltage / Z_Y) * np.sin(omega * t_val - 2 * np.pi / 3 - phi_Y)
            Ib = (voltage / Z_B) * np.sin(omega * t_val - 4 * np.pi / 3 - phi_B)
            calc_details.update({
                "Instantaneous Time (s)": t_val,
                "Voltage R (V)": Vr,
                "Voltage Y (V)": Vy,
                "Voltage B (V)": Vb,
                "Current R (A)": Ir,
                "Current Y (A)": Iy,
                "Current B (A)": Ib
            })
        return jsonify(calc_details)
    except Exception as e:
        app.logger.error(f"Error in calculations_unbalanced: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
