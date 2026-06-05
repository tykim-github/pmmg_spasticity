import os
import sys
import time
import traceback
from collections import deque

import numpy as np
import serial
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QLineEdit, QLabel, QHBoxLayout, QFileDialog,
                             QFormLayout, QMessageBox, QStackedWidget)
from PyQt5.QtCore import QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap

import pyqtgraph as pg
pg.setConfigOptions(antialias=True, foreground='k', background='w')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.gridspec import GridSpec

import serial.tools.list_ports
from numpy.linalg import norm
import csv
from playsound import playsound
import threading

# to make this exe
# pyinstaller --onefile --noconsole --strip --icon=legmus.ico --add-data="files;files" pmmg_receiver_gui_v2.py

PROGRAM_VERSION = "1.07.0"
# 1.01   Initial release
# 1.02   After 2 child patients
# 1.02.2 Minor bug change
# 1.03.1 Add features: Flag added
# 1.03.2 Add features: Flag modified - more accurate flag
# 1.04.1 Add features: Load setting, csv load is now able
# 1.04.2 Reduce size, and csv load will now also plot the flags
# 1.04.3 "LOAD DATA" 버튼에서 "All files (*)" 옵션이 가장 먼저 표기되도록 변환
# 1.05.1 밴드 위/아래의 종아리 둘레를 모두 입력하도록 함
# 1.06.1 Audio added for 102, 103, 104, 105 state
# 1.06.2 Set baudrate 115200 -> 951200
# 1.07.0 Real-time rolling plots during zeroing & recording; review plot on stop

RT_TIMER_MS        = 40    # ms – timer interval for live plot refresh (~25 fps)
ZEROING_WINDOW_S   = 5.0   # seconds of data in the zeroing rolling window
RECORDING_WINDOW_S = 10.0  # seconds of data in the recording rolling window

class DataProcessor:
    def __init__(self, initial_knee_angle=None, initial_ankle_angle=None):
        self.initial_knee_angle = initial_knee_angle
        self.initial_ankle_angle = initial_ankle_angle
        self.q_ti = None
        self.q_si = None
        self.q_fi = None
        self.data_buffer = []

        # Attributes to store processed data
        self.Time = None
        self.knee_angle = None
        self.ankle_angle = None
        self.Pressure = None
        self.knee_flag = None
        self.ankle_flag = None

    def initialize_from_header(self, header_data):
        """헤더 데이터에서 초기 각도를 설정합니다."""
        self.initial_knee_angle = header_data.get('initial_knee_angle')
        self.initial_ankle_angle = header_data.get('initial_ankle_angle')
        self.q_ti = header_data.get('q_ti')
        self.q_si = header_data.get('q_si')
        self.q_fi = header_data.get('q_fi')

    def ensure_initialized(self):
        """Ensure initial quaternion/angle states exist to avoid runtime crashes."""
        if self.q_ti is None:
            self.q_ti = np.array([1.0, 0.0, 0.0, 0.0])
        if self.q_si is None:
            self.q_si = np.array([1.0, 0.0, 0.0, 0.0])
        if self.q_fi is None:
            self.q_fi = np.array([1.0, 0.0, 0.0, 0.0])
        if self.initial_knee_angle is None:
            self.initial_knee_angle = 0.0
        if self.initial_ankle_angle is None:
            self.initial_ankle_angle = 0.0

    def process_row(self, row):
        """Process a single raw data row (list/array of floats).
        Returns (time_ms, knee_angle_deg, ankle_angle_deg, pressure).
        """
        self.ensure_initialized()
        t_ms     = float(row[1])
        q_thigh  = np.asarray(row[2:6],   dtype=float)
        q_shank  = np.asarray(row[6:10],  dtype=float)
        q_foot   = np.asarray(row[10:14], dtype=float)
        pressure = float(row[14])

        q_k = Quaternion.mult(
            Quaternion.mult(q_shank, Quaternion.conj(self.q_si)),
            Quaternion.mult(self.q_ti, Quaternion.conj(q_thigh))
        )
        q_a = Quaternion.mult(
            Quaternion.mult(q_foot, Quaternion.conj(self.q_fi)),
            Quaternion.mult(self.q_si, Quaternion.conj(q_shank))
        )

        knee_deg  = float(np.degrees(Quaternion.angle(q_k)))  + self.initial_knee_angle
        ankle_deg = float(np.degrees(Quaternion.angle(q_a))) + self.initial_ankle_angle
        return t_ms, knee_deg, ankle_deg, pressure

    def process_data(self, data):
        """데이터를 처리하여 결과를 반환합니다."""
        self.ensure_initialized()
        self.Time = data[:, 1]  # Time을 객체 속성으로 저장
        Quaternions = data[:, 2:14]
        self.Pressure = data[:, 14]  # Pressure도 객체 속성으로 저장
        q_thigh = data[:, 2:6]
        q_shank = data[:, 6:10]
        q_foot = data[:, 10:14]

        q_k = np.array([Quaternion.mult(Quaternion.mult(q_shank[i], Quaternion.conj(self.q_si)),
                        Quaternion.mult(self.q_ti, Quaternion.conj(q_thigh[i])))
                        for i in range(len(data))])

        q_a = np.array([Quaternion.mult(Quaternion.mult(q_foot[i], Quaternion.conj(self.q_fi)),
                        Quaternion.mult(self.q_si, Quaternion.conj(q_shank[i])))
                        for i in range(len(data))])

        self.knee_angle = np.degrees([Quaternion.angle(q) for q in q_k]) + self.initial_knee_angle  # knee_angle 속성으로 저장
        self.ankle_angle = np.degrees([Quaternion.angle(q) for q in q_a]) + self.initial_ankle_angle  # ankle_angle 속성으로 저장

        length = len(self.knee_angle)
        if self.knee_flag is None or len(self.knee_flag) != length:
            self.knee_flag = np.zeros(length, dtype=int)
        if self.ankle_flag is None or len(self.ankle_flag) != length:
            self.ankle_flag = np.zeros(length, dtype=int)

        return self.Time, Quaternions, self.Pressure, self.knee_angle, self.ankle_angle

    def calculate_initial_state(self):
        """초기 상태를 계산하여 쿼터니언을 설정합니다."""
        if not self.data_buffer:
            return
        data = np.array([list(map(float, x.split(','))) for x in self.data_buffer])
        Quaternions = data[:, 2:14]

        Quaternions /= norm(Quaternions, axis=1)[:, np.newaxis]
        avg_quaternions = np.mean(Quaternions, axis=0)

        self.q_ti = avg_quaternions[:4]
        self.q_si = avg_quaternions[4:8]
        self.q_fi = avg_quaternions[8:12]

        self.initial_quaternions = (self.q_ti, self.q_si, self.q_fi)

class FileHandler:
    def __init__(self, filename_prefix):
        self.filename_prefix = filename_prefix
        self.session_index = 1
        self.file = None

    def open_new_file(self, header_data):
        """Open a new file with the specified header data."""
        if self.file is not None:
            return  # Avoid opening a new file if one is already open
        filename = f"{self.filename_prefix}_{str(self.session_index).zfill(2)}.txt"
        self.file = open(filename, "w")
        # Save header data as key-value pairs
        for key, value in header_data.items():
            value_str = ','.join(map(str, value)) if isinstance(value, (list, np.ndarray)) else str(value)
            self.file.write(f"{key}={value_str}\n")
        self.file.write("\n")  # Add a blank line to separate the header from the body
        self.session_index += 1

    def write_line(self, line):
        """Write a single line to the file."""
        if self.file:
            self.file.write(line + "\n")

    def close_file(self):
        """Close the currently open file."""
        if self.file:
            self.file.close()
            self.file = None


class SerialReader(QThread):
    data_processed = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    line_read = pyqtSignal(str)
    state_changed = pyqtSignal(str)
    initial_angles_calculated = pyqtSignal(float, float)

    def __init__(self, filename, header_info, processor, parent=None):
        super().__init__(parent)
        self.file_handler = FileHandler(filename)
        self.header_info = header_info
        self.processor = processor
        self.filename = filename
        self.running = False
        self.collecting = False
        self.data_buffer = []
        self.rt_buffer = deque()  # pre-parsed float rows for real-time GUI polling

    def run(self):
        """Main loop for reading serial data."""
        self.running = True
        self.com_port = None

        for port in serial.tools.list_ports.comports():
            if "wch.cn" in port.manufacturer:
                self.com_port = port.device
                break  # USB 포트를 찾으면 루프를 종료합니다.

        if self.com_port is None:
            self.state_changed.emit("SerialFail")
            return  # 메서드를 종료하여 더 이상 코드를 실행하지 않도록 합니다.

        try:
            ser = serial.Serial(self.com_port, 921600, timeout=1)
        except serial.SerialException as e:
            self.state_changed.emit("SerialFail")
            return  # 메서드를 종료하여 더 이상 코드를 실행하지 않도록 합니다.

        try:
            while self.running:
                try:
                    line = ser.readline().decode('utf-8').strip()
                except (serial.SerialException, UnicodeDecodeError):
                    continue  # Skip invalid lines

                if line:
                    if ',' not in line:
                        continue
                    state, _ = line.split(',', 1)
                    if state != "0":
                        self.state_changed.emit(state)
                    if state == "102":
                        play_audio('audio_init_start.mp3')
                        self.processor.data_buffer = []  # Reset buffer
                        self.rt_buffer.clear()
                        self.collecting = True
                    elif state == "103" and self.collecting:
                        play_audio('audio_init_done.mp3')
                        self.processor.calculate_initial_state()
                        self.collecting = False
                    elif state == "104":
                        play_audio('audio_record_start.mp3')
                        self.processor.data_buffer = []  # Reset buffer
                        self.rt_buffer.clear()
                        header_data = {
                            'q_ti': self.processor.q_ti,
                            'q_si': self.processor.q_si,
                            'q_fi': self.processor.q_fi,
                            'initial_knee_angle': self.processor.initial_knee_angle,
                            'initial_ankle_angle': self.processor.initial_ankle_angle,
                            **self.header_info  # Add patient info to header
                        }
                        self.file_handler.open_new_file(header_data)
                        self.collecting = True
                    elif state == "105" and self.collecting:
                        play_audio('audio_record_done.mp3')
                        if self.processor.data_buffer:
                            data = np.array([list(map(float, x.split(','))) for x in self.processor.data_buffer])
                            Time, Quaternions, self.processor.Pressure, knee_angle, ankle_angle = self.processor.process_data(data)
                            self.data_processed.emit(Time, Quaternions, self.processor.Pressure, knee_angle, ankle_angle)
                        self.collecting = False
                        self.file_handler.close_file()
                    elif self.collecting:
                        try:
                            self.rt_buffer.append(list(map(float, line.split(','))))
                        except ValueError:
                            pass
                        self.processor.data_buffer.append(line)
                        if state not in {"102", "103", "104", "105"}:
                            self.file_handler.write_line(line)
        finally:
            self.file_handler.close_file()
            if ser.is_open:
                ser.close()

    def stop(self):
        """Stop the serial reading thread."""
        self.running = False

def lowpass_filter(data, cutoff_freq, fs, order=2):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq

    if order == 2:
        C = 1 / np.tan(np.pi * normal_cutoff)
        a0 = 1 + np.sqrt(2) * C + C**2
        b0 = 1 / a0
        b1 = 2 / a0
        b2 = 1 / a0
        a1 = 2 * (1 - C**2) / a0
        a2 = (1 - np.sqrt(2) * C + C**2) / a0

        # 계수를 배열로 저장
        b = np.array([b0, b1, b2])
        a = np.array([1, a1, a2])  # a0는 1로 정규화
    else:
        raise ValueError("This function currently supports only order=2.")

    # 필터 적용 (전향 및 후향 필터링)
    filtered_data = np.zeros_like(data)
    filtered_data[0] = b[0] * data[0]
    filtered_data[1] = b[0] * data[1] + b[1] * data[0] - a[1] * filtered_data[0]

    for i in range(2, len(data)):
        filtered_data[i] = (b[0] * data[i] + b[1] * data[i - 1] + b[2] * data[i - 2]
                            - a[1] * filtered_data[i - 1] - a[2] * filtered_data[i - 2])

    return filtered_data


def resource_path(relative_path):
    """ PyInstaller로 빌드된 파일 내에서 리소스 파일 경로를 찾는 함수 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        # Use script directory so resources resolve regardless of launch cwd.
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(base_path, relative_path)
    
def play_audio(file_name):
    """비동기적으로 폴더 안에 있는 음성 파일 재생"""
    audio_file = resource_path(os.path.join('files', file_name))
    if not os.path.exists(audio_file):
        return

    def _play_safe(path):
        try:
            playsound(path)
        except Exception:
            # Prevent background audio failures from crashing/log-spamming threads.
            return

    threading.Thread(target=_play_safe, args=(audio_file,), daemon=True).start()

class Quaternion:
    @staticmethod
    def mult(q1, q2):
        """Multiply two quaternions."""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    @staticmethod
    def conj(q):
        """Return the conjugate of a quaternion."""
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    @staticmethod
    def angle(q):
        """Return the angle of a quaternion."""
        w = np.clip(q[0], -1.0, 1.0)  # clamp to avoid NaN from floating-point drift
        return 2 * np.arccos(w)



class SerialDataSaver(QWidget):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.processor = DataProcessor()
        self.display_mode = "idle"   # idle | zeroing | recording | review
        self._session_start = 0.0

        # Rolling buffers – fixed-capacity deques for O(1) append/drop
        _z_max = int(ZEROING_WINDOW_S   * 220)  # 10 % headroom over 200 Hz
        _r_max = int(RECORDING_WINDOW_S * 220)
        self._zeroing_t = deque(maxlen=_z_max)   # seconds from session start
        self._zeroing_q = deque(maxlen=_z_max)   # 12-element lists
        self._zeroing_p = deque(maxlen=_z_max)
        self._t0_data   = None                   # raw ms timestamp of first row

        self._rec_t        = deque(maxlen=_r_max)  # seconds from recording start
        self._rec_t0       = None                  # ms timestamp offset
        self._rec_knee     = deque(maxlen=_r_max)
        self._rec_ankle    = deque(maxlen=_r_max)
        self._rec_pressure = deque(maxlen=_r_max)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._on_timer_tick)
        self.init_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def init_ui(self):
        try:
            screen        = QApplication.primaryScreen()
            geo           = screen.availableGeometry()
            screen_width  = geo.width()
            screen_height = geo.height()

            self._base_font_size = max(10, screen_height // 80)
            self._mpl_font_size  = max(8,  screen_height // 120)

            plt.rcParams['figure.dpi']     = screen.logicalDotsPerInch()
            plt.rcParams['font.size']       = self._mpl_font_size
            plt.rcParams['axes.labelsize']  = self._mpl_font_size
            plt.rcParams['xtick.labelsize'] = self._mpl_font_size
            plt.rcParams['ytick.labelsize'] = self._mpl_font_size
            plt.rcParams['legend.fontsize'] = self._mpl_font_size

            self.setGeometry(
                int(screen_width * 0.15), int(screen_height * 0.1),
                int(screen_width * 0.7),  int(screen_height * 0.8),
            )

            layout = QHBoxLayout()

            # ── Left: plot area (pyqtgraph for live; matplotlib for review) ─
            plot_layout = QVBoxLayout()

            self.plot_stack = QStackedWidget()

            # Index 0: pyqtgraph – live zeroing / recording
            self.pg_widget = pg.GraphicsLayoutWidget()
            self.plot_stack.addWidget(self.pg_widget)

            # Index 1: matplotlib – review / idle
            mpl_container = QWidget()
            mpl_layout = QVBoxLayout(mpl_container)
            mpl_layout.setContentsMargins(0, 0, 0, 0)
            self.fig    = plt.figure(figsize=(15, 15))
            self.canvas = FigureCanvas(self.fig)
            self.canvas.mpl_connect('button_press_event', self.on_click)
            self.toolbar = NavigationToolbar(self.canvas, self)
            mpl_layout.addWidget(self.toolbar)
            mpl_layout.addWidget(self.canvas)
            self.plot_stack.addWidget(mpl_container)

            self.plot_stack.setCurrentIndex(1)   # start in matplotlib (idle/review)
            plot_layout.addWidget(self.plot_stack)

            # Default idle layout matches the review layout (3 subplots)
            self.axes = [self.fig.add_subplot(3, 1, i + 1) for i in range(3)]

            version_label = QLabel(f"Spasticity Measurement Software v{PROGRAM_VERSION}")
            version_label.setAlignment(Qt.AlignCenter)
            version_label.setStyleSheet(
                f"font-size: {int(self._base_font_size * 0.7)}px; color: gray;")
            plot_layout.addWidget(version_label)
            layout.addLayout(plot_layout, 2)

            # ── Right: controls ───────────────────────────────────────────
            control_layout = QVBoxLayout()

            self.status_label = QLabel("Status: Idle")
            self.status_label.setAlignment(Qt.AlignCenter)
            control_layout.addWidget(self.status_label)
            self._set_status("Idle", "#555555")

            self.elapsed_label = QLabel("")
            self.elapsed_label.setAlignment(Qt.AlignCenter)
            self.elapsed_label.setStyleSheet(
                f"font-size: {self._base_font_size}px; color: #333333;")
            control_layout.addWidget(self.elapsed_label)

            # Reference image
            self.image_label = QLabel(self)
            image_path = resource_path(os.path.join('files', 'joint_angle_definition.png'))
            pixmap = QPixmap(image_path)
            max_w = int(screen_width * 0.3)
            max_h = int(screen_height * 0.25)
            if not pixmap.isNull():
                self.image_label.setPixmap(
                    pixmap.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.setAlignment(Qt.AlignCenter)
            control_layout.addWidget(self.image_label)

            # Patient information form
            patient_form = QFormLayout()
            fs = self._base_font_size

            def _field():
                w = QLineEdit(self)
                w.setStyleSheet(f"font-size: {fs}px;")
                return w

            self.filename_input                   = _field()
            self.patient_shank_upper_circum_input = _field()
            self.patient_shank_lower_circum_input = _field()
            self.patient_band_elongation_input    = _field()
            self.initial_knee_angle_input         = _field()
            self.initial_ankle_angle_input        = _field()
            self.recorder_name_input              = _field()

            patient_form.addRow("Trial Name:",                      self.filename_input)
            patient_form.addRow("Shank Upper Circumference (mm):",  self.patient_shank_upper_circum_input)
            patient_form.addRow("Shank Lower Circumference (mm):",  self.patient_shank_lower_circum_input)
            patient_form.addRow("Elongated Band Length (mm):",      self.patient_band_elongation_input)
            patient_form.addRow("Initial Knee Angle (deg):",        self.initial_knee_angle_input)
            patient_form.addRow("Initial Ankle Angle (deg):",       self.initial_ankle_angle_input)
            patient_form.addRow("Recorder Name:",                   self.recorder_name_input)
            control_layout.addLayout(patient_form)

            self.initial_angles_label = QLabel("")
            self.initial_angles_label.setAlignment(Qt.AlignCenter)
            self.initial_angles_label.setStyleSheet(
                f"font-size: {int(fs * 0.8)}px; color: gray;")
            control_layout.addWidget(self.initial_angles_label)

            # Buttons
            btn_style = f"font-size: {int(fs * 1.4)}px;"
            btns = QVBoxLayout()

            self.start_btn = QPushButton('START', self)
            self.start_btn.setStyleSheet(btn_style)
            self.start_btn.clicked.connect(self.start_reading)
            btns.addWidget(self.start_btn)

            self.end_btn = QPushButton('END', self)
            self.end_btn.setStyleSheet(btn_style)
            self.end_btn.clicked.connect(self.close_app)
            btns.addWidget(self.end_btn)

            self.load_btn = QPushButton('LOAD DATA', self)
            self.load_btn.setStyleSheet(btn_style)
            self.load_btn.clicked.connect(self.load_data)
            btns.addWidget(self.load_btn)

            load_setting_btn = QPushButton('LOAD SETTING', self)
            load_setting_btn.setStyleSheet(btn_style)
            load_setting_btn.clicked.connect(self.load_setting)
            btns.addWidget(load_setting_btn)

            self.save_plot_btn = QPushButton('SAVE PLOT', self)
            self.save_plot_btn.setStyleSheet(btn_style)
            self.save_plot_btn.clicked.connect(self.save_plot)
            self.save_plot_btn.setEnabled(False)
            btns.addWidget(self.save_plot_btn)

            self.export_csv_btn = QPushButton('EXPORT CSV', self)
            self.export_csv_btn.setStyleSheet(btn_style)
            self.export_csv_btn.clicked.connect(self.export_csv)
            self.export_csv_btn.setEnabled(False)
            btns.addWidget(self.export_csv_btn)

            control_layout.addLayout(btns)
            layout.addLayout(control_layout, 1)

            self.setLayout(layout)
            self.setWindowTitle('Spasticity Measurement Software')
            self.show()
        except Exception as e:
            self.handle_exception(e)

    # ── Status helpers ───────────────────────────────────────────────────────

    def _set_status(self, message, bg_color):
        light_bgs = {"#c8860a", "goldenrod"}
        text_color = "black" if bg_color in light_bgs else "white"
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(
            f"font-size: {int(self._base_font_size * 1.4)}px; font-weight: bold; "
            f"background-color: {bg_color}; color: {text_color}; "
            f"padding: 4px; border-radius: 4px;"
        )

    # ── Display-mode transitions ─────────────────────────────────────────────

    def _enter_zeroing_mode(self):
        self.display_mode = "zeroing"
        self._zeroing_t.clear(); self._zeroing_q.clear(); self._zeroing_p.clear()
        self._t0_data = None
        self._session_start = time.monotonic()

        self.plot_stack.setCurrentIndex(0)
        self.pg_widget.clear()

        comp_labels = ['w', 'x', 'y', 'z']
        imu_titles  = ['Thigh IMU (quaternion)', 'Shank IMU (quaternion)', 'Foot IMU (quaternion)']
        pg_colors   = ['#222222', '#e74c3c', '#27ae60', '#2980b9']

        self._pg_qplots  = []
        self._pg_qcurves = []   # [imu][comp]
        for i, title in enumerate(imu_titles):
            p = self.pg_widget.addPlot(row=i, col=0, title=title)
            p.setLabel('left', 'Component')
            p.setYRange(-1.1, 1.1)
            p.disableAutoRange('y')
            p.showGrid(x=True, y=True, alpha=0.3)
            p.addLegend(offset=(1, 0))
            curves = [p.plot(pen=pg.mkPen(c, width=1), name=lbl)
                      for c, lbl in zip(pg_colors, comp_labels)]
            self._pg_qplots.append(p)
            self._pg_qcurves.append(curves)

        self._pg_pplot = self.pg_widget.addPlot(row=3, col=0, title='Pressure')
        self._pg_pplot.setLabel('left', 'Pressure (kPa)')
        self._pg_pplot.setLabel('bottom', 'Time (s)')
        self._pg_pplot.showGrid(x=True, y=True, alpha=0.3)
        self._pg_pcurve = self._pg_pplot.plot(pen=pg.mkPen('k', width=1))

        self.update_timer.start(RT_TIMER_MS)

    def _exit_zeroing_mode(self):
        self.update_timer.stop()
        self.elapsed_label.setText("")
        self.display_mode = "idle"

    def _enter_recording_mode(self):
        self.display_mode = "recording"
        self._rec_t.clear(); self._rec_t0 = None
        self._rec_knee.clear(); self._rec_ankle.clear(); self._rec_pressure.clear()
        self._session_start = time.monotonic()

        self.plot_stack.setCurrentIndex(0)
        self.pg_widget.clear()

        self._pg_angle_plot = self.pg_widget.addPlot(row=0, col=0, title='Joint Angle')
        self._pg_angle_plot.setLabel('left', 'Angle (deg)')
        self._pg_angle_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pg_angle_plot.addLegend(offset=(1, 0))
        self._pg_knee_curve  = self._pg_angle_plot.plot(pen=pg.mkPen('#8b0000', width=1), name='Knee')
        self._pg_ankle_curve = self._pg_angle_plot.plot(pen=pg.mkPen('#00008b', width=1), name='Ankle')

        self._pg_vel_plot = self.pg_widget.addPlot(row=1, col=0, title='Angular Velocity')
        self._pg_vel_plot.setLabel('left', 'Velocity (deg/s)')
        self._pg_vel_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pg_vel_plot.addLegend(offset=(1, 0))
        self._pg_kvel_curve = self._pg_vel_plot.plot(pen=pg.mkPen('#8b0000', width=1), name='Knee')
        self._pg_avel_curve = self._pg_vel_plot.plot(pen=pg.mkPen('#00008b', width=1), name='Ankle')

        self._pg_pres_plot = self.pg_widget.addPlot(row=2, col=0, title='Pressure')
        self._pg_pres_plot.setLabel('left', 'Pressure (kPa)')
        self._pg_pres_plot.setLabel('bottom', 'Time (s)')
        self._pg_pres_plot.showGrid(x=True, y=True, alpha=0.3)
        self._pg_prec_curve = self._pg_pres_plot.plot(pen=pg.mkPen('k', width=1))

        self.update_timer.start(RT_TIMER_MS)

    def _stop_recording(self):
        self.update_timer.stop()
        self.elapsed_label.setText("")
        self.display_mode = "review_pending"

    # ── QTimer callback ──────────────────────────────────────────────────────

    def _on_timer_tick(self):
        if self.thread is None or not self.thread.isRunning():
            return

        new_rows = []
        buf = self.thread.rt_buffer
        while buf:
            try:
                new_rows.append(buf.popleft())
            except IndexError:
                break

        elapsed = time.monotonic() - self._session_start

        if self.display_mode == "zeroing":
            self.elapsed_label.setText(f"Zeroing: {elapsed:.1f} s")
            if new_rows:
                self._update_zeroing_plot(new_rows)
        elif self.display_mode == "recording":
            self.elapsed_label.setText(f"Recording: {elapsed:.1f} s")
            if new_rows:
                self._update_recording_plot(new_rows)

    def _update_zeroing_plot(self, new_rows):
        for row in new_rows:
            if len(row) < 15:
                continue
            t_ms = row[1]
            if self._t0_data is None:
                self._t0_data = t_ms
            self._zeroing_t.append((t_ms - self._t0_data) / 1000.0)
            self._zeroing_q.append(row[2:14])
            self._zeroing_p.append(row[14])

        if not self._zeroing_t:
            return

        t_arr = np.array(self._zeroing_t)
        q_arr = np.array(self._zeroing_q)   # (N, 12)
        p_arr = np.array(self._zeroing_p)

        t_max = t_arr[-1]
        t_min = max(t_max - ZEROING_WINDOW_S, t_arr[0])
        mask  = t_arr >= t_min
        t_w, q_w, p_w = t_arr[mask], q_arr[mask], p_arr[mask]

        for imu, curves in enumerate(self._pg_qcurves):
            base = imu * 4
            for comp, curve in enumerate(curves):
                curve.setData(t_w, q_w[:, base + comp])
        self._pg_pcurve.setData(t_w, p_w)

        for p in self._pg_qplots:
            p.setXRange(t_min, t_max + 0.05, padding=0)
        self._pg_pplot.setXRange(t_min, t_max + 0.05, padding=0)

    def _update_recording_plot(self, new_rows):
        for row in new_rows:
            if len(row) < 15:
                continue
            try:
                t_ms, knee, ankle, pressure = self.processor.process_row(row)
                if self._rec_t0 is None:
                    self._rec_t0 = t_ms
                self._rec_t.append((t_ms - self._rec_t0) / 1000.0)
                self._rec_knee.append(knee)
                self._rec_ankle.append(ankle)
                self._rec_pressure.append(pressure)
            except Exception:
                continue

        n = len(self._rec_t)
        if n < 2:
            return

        t_arr = np.array(self._rec_t)
        k_arr = np.array(self._rec_knee)
        a_arr = np.array(self._rec_ankle)
        p_arr = np.array(self._rec_pressure)

        t_max = t_arr[-1]
        t_min = max(t_max - RECORDING_WINDOW_S, t_arr[0])
        mask  = t_arr >= t_min
        t_w = t_arr[mask]; k_w = k_arr[mask]
        a_w = a_arr[mask]; p_w = p_arr[mask]

        self._pg_knee_curve.setData(t_w, k_w)
        self._pg_ankle_curve.setData(t_w, a_w)

        if len(k_w) >= 4:
            dt = 0.005
            kv_raw = np.gradient(k_w, dt)
            av_raw = np.gradient(a_w, dt)
            win = min(21, len(kv_raw))
            kernel = np.ones(win) / win
            self._pg_kvel_curve.setData(t_w, np.convolve(kv_raw, kernel, mode='same'))
            self._pg_avel_curve.setData(t_w, np.convolve(av_raw, kernel, mode='same'))

        self._pg_prec_curve.setData(t_w, p_w)

        for p in (self._pg_angle_plot, self._pg_vel_plot, self._pg_pres_plot):
            p.setXRange(t_min, t_max + 0.05, padding=0)

    # ── Serial thread management ─────────────────────────────────────────────

    def handle_exception(self, e):
        with open("error_log.txt", "a") as f:
            traceback.print_exc(file=f)
        QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def start_reading(self):
        if self.thread and self.thread.isRunning():
            return
        try:
            trial_name        = self.filename_input.text().strip()
            shank_upper       = self.patient_shank_upper_circum_input.text()
            shank_lower       = self.patient_shank_lower_circum_input.text()
            band_elongation   = self.patient_band_elongation_input.text()
            initial_knee_deg  = float(self.initial_knee_angle_input.text())
            initial_ankle_deg = float(self.initial_ankle_angle_input.text())
            recorder_name     = self.recorder_name_input.text()

            if not trial_name:
                QMessageBox.warning(self, "Missing input", "Please enter a Trial Name.")
                return

            self.file_name = trial_name
            self.start_btn.setEnabled(False)
            self.load_btn.setEnabled(False)
            self.filename_input.setEnabled(False)
            self.save_plot_btn.setEnabled(False)
            self.export_csv_btn.setEnabled(False)
            self.setWindowTitle(f"Spasticity Measurement \u2014 {trial_name}")

            header_info = {
                "Trial Name":                      trial_name,
                "Shank Upper Circumference (mm)":  shank_upper,
                "Shank Lower Circumference (mm)":  shank_lower,
                "Elongated Band Length(mm)":        band_elongation,
                "Initial Knee Angle (deg)":         initial_knee_deg,
                "Initial Ankle Angle (deg)":        initial_ankle_deg,
                "Recorder Name":                    recorder_name,
            }

            self.processor = DataProcessor(initial_knee_deg, initial_ankle_deg)

            self.thread = SerialReader(trial_name, header_info, self.processor, self)
            self.thread.data_processed.connect(self.plot_data)
            self.thread.line_read.connect(self.handle_line_read)
            self.thread.state_changed.connect(self.update_status_label)
            self.thread.initial_angles_calculated.connect(self.display_initial_angles)
            self.thread.finished.connect(self.on_thread_finished)
            self.thread.start()
            self._set_status("Waiting for device\u2026", "#555555")

        except ValueError:
            QMessageBox.warning(self, "Invalid input",
                                "Initial knee and ankle angles must be numbers.")
        except Exception as e:
            self.handle_exception(e)

    def update_status_label(self, status_code):
        STATUS = {
            "101": ("Standby",                   "#555555"),
            "102": ("Leg zeroing started",        "#c8860a"),
            "103": ("Leg zeroing done",           "#555555"),
            "104": ("Recording started",          "#196f3d"),
            "105": ("Recording stopped",          "#555555"),
            "106": ("Magnetometer calibrating",   "#c8860a"),
            "201": ("ERROR \u2013 Low voltage",         "crimson"),
            "202": ("ERROR \u2013 pMMG malfunction",    "crimson"),
            "203": ("ERROR \u2013 IMU malfunction",     "crimson"),
            "SerialFail": ("ERROR \u2013 No USB port",  "crimson"),
        }
        msg, color = STATUS.get(status_code, ("ERROR \u2013 ???", "crimson"))
        self._set_status(msg, color)

        if status_code == "102" and self.display_mode != "zeroing":
            self._enter_zeroing_mode()
        elif status_code == "103" and self.display_mode == "zeroing":
            self._exit_zeroing_mode()
        elif status_code == "104" and self.display_mode != "recording":
            self._enter_recording_mode()
        elif status_code == "105" and self.display_mode == "recording":
            self._stop_recording()

    def on_thread_finished(self):
        self.update_timer.stop()
        self.elapsed_label.setText("")
        self.display_mode = "idle"
        self.start_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.filename_input.setEnabled(True)
        self.setWindowTitle("Spasticity Measurement Software")
        self._set_status("Idle", "#555555")

    def handle_line_read(self, line):
        pass

    def display_initial_angles(self, knee_angle, ankle_angle):
        self.initial_angles_label.setText(
            f"Initial Knee: {knee_angle:.2f}\u00b0     Initial Ankle: {ankle_angle:.2f}\u00b0")

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_data(self):
        try:
            options = QFileDialog.Options()
            self.file_name, _ = QFileDialog.getOpenFileName(
                self, "Load Data File", "",
                "All Files (*);;Text Files (*.txt);;CSV Files (*.csv)",
                options=options,
            )
            if not self.file_name:
                return

            if self.file_name.lower().endswith('.txt'):
                header_data = {}
                with open(self.file_name, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            break
                        key, value_str = line.split('=', 1)
                        try:
                            vals = list(map(float, value_str.split(',')))
                            header_data[key] = np.array(vals) if len(vals) > 1 else vals[0]
                        except ValueError:
                            header_data[key] = value_str

                self.processor.initialize_from_header(header_data)
                data = np.loadtxt(self.file_name, delimiter=',',
                                  skiprows=len(header_data) + 1)
                Time, Quats, Pressure, knee, ankle = self.processor.process_data(data)
                self.plot_data(Time, Quats, Pressure, knee, ankle)

            elif self.file_name.lower().endswith('.csv'):
                with open(self.file_name, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    data   = list(reader)

                if not data:
                    QMessageBox.warning(self, "File Error", "CSV 파일에 데이터가 없습니다.")
                    return

                try:
                    data = np.array(data, dtype=float)
                except ValueError:
                    QMessageBox.warning(self, "File Error",
                                        "CSV 파일에 숫자가 아닌 값이 포함되어 있습니다.")
                    return

                required = ['Time_sec', 'knee_angle', 'ankle_angle', 'Pressure']
                for col in required:
                    if col not in header:
                        QMessageBox.warning(self, "File Error",
                                            f"CSV 파일에 필수 열 '{col}'이(가) 없습니다.")
                        return

                ti = header.index('Time_sec');   ki = header.index('knee_angle')
                ai = header.index('ankle_angle'); pi = header.index('Pressure')

                Time_sec    = data[:, ti]
                knee_angle  = data[:, ki]
                ankle_angle = data[:, ai]
                Pressure    = data[:, pi]

                self.processor.knee_flag = (
                    data[:, header.index('knee_flag')].astype(int)
                    if 'knee_flag' in header
                    else np.zeros(len(knee_angle), dtype=int)
                )
                self.processor.ankle_flag = (
                    data[:, header.index('ankle_flag')].astype(int)
                    if 'ankle_flag' in header
                    else np.zeros(len(ankle_angle), dtype=int)
                )
                self.processor.Time        = Time_sec * 1000
                self.processor.knee_angle  = knee_angle
                self.processor.ankle_angle = ankle_angle
                self.processor.Pressure    = Pressure

                self.plot_data(self.processor.Time, None,
                               self.processor.Pressure, knee_angle, ankle_angle)
            else:
                QMessageBox.warning(self, "File Error",
                                    "지원하지 않는 파일 형식입니다. txt 또는 csv 파일을 선택하십시오.")
        except Exception as e:
            self.handle_exception(e)

    def load_setting(self):
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Load Setting File", "",
                "Text Files (*.txt);;All Files (*)", options=options,
            )
            if not file_name:
                return

            header_data = {}
            with open(file_name, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        break
                    key, value_str = line.split('=', 1)
                    header_data[key] = value_str

            self.filename_input.setText(header_data.get('Trial Name', ''))
            self.patient_shank_upper_circum_input.setText(
                header_data.get('Shank Upper Circumference (mm)', ''))
            self.patient_shank_lower_circum_input.setText(
                header_data.get('Shank Lower Circumference (mm)', ''))
            self.patient_band_elongation_input.setText(
                header_data.get('Elongated Band Length(mm)', ''))
            self.initial_knee_angle_input.setText(
                header_data.get('Initial Knee Angle (deg)', ''))
            self.initial_ankle_angle_input.setText(
                header_data.get('Initial Ankle Angle (deg)', ''))
            self.recorder_name_input.setText(header_data.get('Recorder Name', ''))
        except Exception as e:
            self.handle_exception(e)

    # ── Review / static plot ──────────────────────────────────────────────────

    def plot_data(self, Time, Quaternions, Pressure, knee_angle, ankle_angle):
        """Full static review plot – called after recording stops or when loading data."""
        self.update_timer.stop()
        self.display_mode = "review"
        self.plot_stack.setCurrentIndex(1)   # switch to matplotlib review page

        self.fig.clf()
        gs = GridSpec(3, 1, figure=self.fig, hspace=0.45)
        self.axes = [self.fig.add_subplot(gs[i]) for i in range(3)]

        self.Time_sec    = Time / 1000.0
        self.knee_angle  = knee_angle
        self.ankle_angle = ankle_angle

        self.line_knee,  = self.axes[0].plot(self.Time_sec, self.knee_angle,
                                              color='darkred',  label='Knee joint',  picker=5)
        self.line_ankle, = self.axes[0].plot(self.Time_sec, self.ankle_angle,
                                              color='darkblue', label='Ankle joint', picker=5)
        self.axes[0].set_xlabel("Time (s)")
        self.axes[0].set_ylabel("Angle (deg)")
        self.axes[0].legend(); self.axes[0].grid(True)

        ki = np.where(self.processor.knee_flag  != 0)[0]
        ai = np.where(self.processor.ankle_flag != 0)[0]
        self.scatter_knee  = self.axes[0].scatter(
            self.Time_sec[ki], self.knee_angle[ki],
            facecolors='none', edgecolors='red',  s=50, label='Flagged Knee')
        self.scatter_ankle = self.axes[0].scatter(
            self.Time_sec[ai], self.ankle_angle[ai],
            facecolors='none', edgecolors='blue', s=50, label='Flagged Ankle')

        dt   = 0.005
        fs_s = 1.0 / dt
        kvel = lowpass_filter(np.diff(knee_angle)  / dt, 5, fs_s)
        avel = lowpass_filter(np.diff(ankle_angle) / dt, 5, fs_s)
        self.axes[1].plot(self.Time_sec[:-1], kvel,
                          color='darkred',  label='Knee velocity (filtered)')
        self.axes[1].plot(self.Time_sec[:-1], avel,
                          color='darkblue', label='Ankle velocity (filtered)')
        self.axes[1].set_xlabel("Time (s)")
        self.axes[1].set_ylabel("Angular velocity (deg/s)")
        self.axes[1].legend(); self.axes[1].grid(True)

        self.axes[2].plot(self.Time_sec, Pressure, color='black', label='pMMG')
        self.axes[2].set_xlabel("Time (s)")
        self.axes[2].set_ylabel("Pressure (kPa)")
        self.axes[2].legend(); self.axes[2].grid(True)

        self.canvas.draw()
        self.save_plot_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.elapsed_label.setText("")

    def on_click(self, event):
        if self.display_mode != "review":
            return
        if not hasattr(self, 'Time_sec') or event.inaxes != self.axes[0]:
            return
        x_click = event.xdata; y_click = event.ydata
        if x_click is None or y_click is None:
            return

        knee_y  = np.interp(x_click, self.Time_sec, self.knee_angle)
        ankle_y = np.interp(x_click, self.Time_sec, self.ankle_angle)
        kd = abs(y_click - knee_y); ad = abs(y_click - ankle_y)
        threshold = 5

        if kd < ad and kd < threshold:
            ind = np.argmin(np.abs(self.Time_sec - x_click))
            self.processor.knee_flag[ind] ^= 1
            ki = np.where(self.processor.knee_flag != 0)[0]
            self.scatter_knee.set_offsets(
                np.c_[self.Time_sec[ki], self.knee_angle[ki]])
            self.canvas.draw_idle()
        elif ad <= kd and ad < threshold:
            ind = np.argmin(np.abs(self.Time_sec - x_click))
            self.processor.ankle_flag[ind] ^= 1
            ai = np.where(self.processor.ankle_flag != 0)[0]
            self.scatter_ankle.set_offsets(
                np.c_[self.Time_sec[ai], self.ankle_angle[ai]])
            self.canvas.draw_idle()

    def save_plot(self):
        try:
            if hasattr(self, 'file_name'):
                self.fig.savefig(os.path.splitext(self.file_name)[0] + '.png')
        except Exception as e:
            self.handle_exception(e)

    def export_csv(self):
        try:
            if not hasattr(self, 'file_name'):
                return
            out  = os.path.splitext(self.file_name)[0] + '.csv'
            dt   = 0.005; fs_s = 1.0 / dt
            kvel = lowpass_filter(np.diff(self.processor.knee_angle)  / dt, 13, fs_s)
            avel = lowpass_filter(np.diff(self.processor.ankle_angle) / dt, 13, fs_s)
            data_out = {
                'Time_sec':               self.processor.Time[:-1] / 1000.0,
                'knee_angle':             self.processor.knee_angle[:-1],
                'ankle_angle':            self.processor.ankle_angle[:-1],
                'knee_velocity_filtered': kvel,
                'ankle_velocity_filtered':avel,
                'Pressure':               self.processor.Pressure[:-1],
                'knee_flag':              self.processor.knee_flag[:-1],
                'ankle_flag':             self.processor.ankle_flag[:-1],
            }
            with open(out, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=data_out.keys())
                writer.writeheader()
                n = len(self.processor.Time) - 1
                for i in range(n):
                    writer.writerow({k: v[i] for k, v in data_out.items()})
        except Exception as e:
            self.handle_exception(e)

    def close_app(self):
        try:
            if self.thread and self.thread.isRunning():
                self.thread.stop()
                self.thread.wait()
            self.update_timer.stop()
            self.start_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.filename_input.setEnabled(True)
        except Exception as e:
            self.handle_exception(e)


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = SerialDataSaver()
        sys.exit(app.exec_())

    except Exception as e:
        with open("error_log.txt", "a") as f:
            traceback.print_exc(file=f)
        QMessageBox.critical(None, "Critical Error",
                             f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
