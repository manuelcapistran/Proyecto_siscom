import sys
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

FS = 44100
BLOCKSIZE = 512

# Bandas con descripciones más claras
BAND_DESCRIPTIONS = [
    (20, 60, "Subgraves – bajos profundos"),
    (60, 250, "Graves – kick / bajo"),
    (250, 1000, "Medios – voz / guitarra"),
    (1000, 4000, "Agudos medios – claridad"),
    (4000, 16000, "Agudos altos – aire / brillo")
]

class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 3))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.x = np.linspace(0, FS/2, BLOCKSIZE//2 + 1)
        self.line, = self.ax.plot(self.x, np.zeros_like(self.x))
        self.ax.set_ylim(-80, 0)
        self.ax.set_xlim(0, FS//2)
        self.ax.set_xlabel('Frecuencia [Hz]')
        self.ax.set_ylabel('dB')
        self.ax.grid(True)

    def update_spectrum(self, audio_block):
        windowed = audio_block * np.hanning(len(audio_block))
        yf = np.fft.rfft(windowed)
        yf_db = 20 * np.log10(np.abs(yf) + 1e-6)
        self.line.set_ydata(yf_db)
        self.draw()

class EqualizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ecualizador 5 Bandas')
        self.setGeometry(100, 100, 1000, 500)
        self.gains = [1.0] * 5
        self.running = False

        self.spectrum_canvas = SpectrumCanvas(self)
        self.last_block = np.zeros(BLOCKSIZE)

        self.init_ui()
        self.init_filters()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_spectrum_gui)

    def init_ui(self):
        main_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        sliders_layout = QHBoxLayout()

        self.sliders = []
        self.labels = []

        for i, (f1, f2, desc) in enumerate(BAND_DESCRIPTIONS):
            group = QGroupBox()
            group_layout = QVBoxLayout()

            # Etiqueta superior con la descripción
            title = QLabel(f"{desc}\n({f1}-{f2} Hz)")
            title.setAlignment(Qt.AlignCenter)
            title.setWordWrap(True)
            title.setStyleSheet("font-weight: bold;")

            # Etiqueta de ganancia
            label = QLabel('Ganancia: 1.00x')
            label.setAlignment(Qt.AlignCenter)

            # Slider
            slider = QSlider(Qt.Vertical)
            slider.setMinimum(50)
            slider.setMaximum(200)
            slider.setValue(100)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksLeft)
            slider.valueChanged.connect(lambda val, idx=i: self.set_gain(idx, val))

            # Agregar al layout
            group_layout.addWidget(title)
            group_layout.addWidget(slider)
            group_layout.addWidget(label)
            group.setLayout(group_layout)

            sliders_layout.addWidget(group)
            self.sliders.append(slider)
            self.labels.append(label)

        self.start_btn = QPushButton('Iniciar')
        self.start_btn.clicked.connect(self.start_eq)
        self.stop_btn = QPushButton('Detener')
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_eq)

        control_layout.addLayout(sliders_layout)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)

        main_layout.addLayout(control_layout, 1)
        main_layout.addWidget(self.spectrum_canvas, 3)
        self.setLayout(main_layout)

    def init_filters(self):
        self.sos_filters = [
            signal.butter(4, [f1/(FS/2), f2/(FS/2)], btype='band', output='sos')
            for f1, f2, _ in BAND_DESCRIPTIONS
        ]
        self.sos_zf = [signal.sosfilt_zi(sos) * 0 for sos in self.sos_filters]

    def set_gain(self, idx, val):
        gain = val / 100.
        self.gains[idx] = gain
        self.labels[idx].setText(f'Ganancia: {gain:.2f}x')

    def audio_callback(self, indata, outdata, frames, time, status):
        x = indata[:, 0]
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x)
        y = np.zeros_like(x)

        for i in range(5):
            sos = self.sos_filters[i]
            x_filt, self.sos_zf[i] = signal.sosfilt(sos, x, zi=self.sos_zf[i])
            safe_gain = min(self.gains[i], 2.0)
            y += safe_gain * x_filt

        peak = np.max(np.abs(y))
        if peak > 1.0:
            y = y / peak * 0.9  # limitador suave

        outdata[:, 0] = y
        self.last_block = y.copy()

    def start_eq(self):
        if not self.running:
            self.running = True
            self.stream = sd.Stream(samplerate=FS,
                                    blocksize=BLOCKSIZE,
                                    channels=1,
                                    dtype='float32',
                                    callback=self.audio_callback)
            self.stream.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.timer.start(50)

    def stop_eq(self):
        if self.running:
            self.running = False
            self.stream.stop()
            self.stream.close()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.timer.stop()

    def update_spectrum_gui(self):
        self.spectrum_canvas.update_spectrum(self.last_block)

    def closeEvent(self, event):
        self.stop_eq()
        event.accept()

def main():
    app = QApplication(sys.argv)
    eq = EqualizerApp()
    eq.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
