import numpy as np
import sounddevice as sd
from scipy.signal import butter, filtfilt
import tkinter as tk

# Configuración general
fs = 48000.0
frame_size = 4096
bands = [(20, 60), (61, 250), (251, 1000), (1001, 4000), (4001, 16000)]
#Estilo en V (realza bajos y agudos)
gains = [3.0, 1.0, 0.5, 1.0, 3.0] # Ganancias iniciales

# Filtros
def butter_bandpass(lowcut, highcut, fs, order=6):  # Orden aumentado
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_filters(data):
    output = np.zeros_like(data)
    for i, (low, high) in enumerate(bands):
        b, a = butter_bandpass(low, high, fs)
        try:
            filtered = filtfilt(b, a, data)
        except ValueError:
            filtered = np.zeros_like(data)
        output += filtered * gains[i]

    # Normalización para evitar clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val

    # Recorte por seguridad
    output = np.clip(output, -1.0, 1.0)

    return output

# Callback de procesamiento
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    mono = indata[:, 0]
    processed = apply_filters(mono)
    outdata[:] = np.column_stack([processed, processed])

# Interfaz gráfica
def create_interface():
    root = tk.Tk()
    root.title("Ecualizador de 5 Bandas")

    labels = [  "Subgraves (20–60Hz) – bajos profundos",
                "Graves (60–250Hz) – kick/bajo",
                "Medios (250Hz–1kHz) – voz, guitarra",
                "Agudos M. (1–4kHz) – claridad",
                "Agudos A. (4–16kHz) – aire, brillo"]

    sliders = []

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0, sticky="w")
        slider = tk.Scale(root, from_=0.0, to=5.0, resolution=0.1, orient="horizontal",
                          length=300, command=lambda val, idx=i: update_gain(idx, val))
        slider.set(gains[i])
        slider.grid(row=i, column=1)
        sliders.append(slider)

    tk.Label(root, text="Cierra esta ventana para detener").grid(row=6, columnspan=2)

    return root

def update_gain(index, value):
    gains[index] = float(value)

# Iniciar todo
def main():
    root = create_interface()

    try:
        with sd.Stream(device=(1,5),  # Mezcla estéreo → Altavoces (WASAPI)
                       channels=(2,2),
                       callback=callback,
                       samplerate=fs,
                       blocksize=frame_size):
            print("Streaming activo. Ajusta con sliders.")
            root.mainloop()

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

tk.Scale