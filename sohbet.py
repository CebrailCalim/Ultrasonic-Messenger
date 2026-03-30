import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import tkinter as tk
from tkinter import scrolledtext
import threading
import queue
import time
from datetime import datetime
import os

# --- PARAMETRELER (11kHz - 12.5kHz) ---
FS = 44100
F1_BASE = 11000 
F2_BASE = 12500 
STEP = 60           
THRESHOLD = 2.5     
DURATION = 0.5      
CONFIDENCE_REQ = 3  
LINE_TIMEOUT = 10   

class SecureChatV14:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Modem v14 - Echo Protection")
        self.root.geometry("1100x700")
        self.root.configure(bg="#000")

        # --- DEĞİŞKENLER ---
        self.msg_queue = queue.Queue()
        self.confidence_buffer = []
        self.last_char_code = -1
        self.last_signal_time = 0
        self.is_block_active = False
        self.is_running = True
        self.is_transmitting = False 

        # --- ARAYÜZ TASARIMI ---
        self.chat_frame = tk.Frame(root, bg="#000")
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(self.chat_frame, bg="#000", fg="#39FF14", 
                                                  font=("Consolas", 12), state='disabled', borderwidth=0)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.controls = tk.Frame(self.chat_frame, bg="#111", pady=5)
        self.controls.pack(fill=tk.X, padx=5, pady=5)

        self.entry = tk.Entry(self.controls, bg="#222", fg="#fff", font=("Arial", 12), insertbackground="white")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entry.bind("<Return>", lambda e: self.send_msg())

        self.send_btn = tk.Button(self.controls, text="GÖNDER", bg="#39FF14", fg="#000", font=("Arial", 10, "bold"), command=self.send_msg)
        self.send_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = tk.Button(self.controls, text="LOG TEMİZLE", bg="#FF3131", fg="#fff", font=("Arial", 10, "bold"), command=self.clear_log)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # SAĞ: Grafik
        self.graph_frame = tk.Frame(root, bg="#000")
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(4, 8))
        self.fig.patch.set_facecolor('#000')
        self.plot_x1 = np.linspace(F1_BASE-100, F1_BASE+700, 300)
        self.plot_x2 = np.linspace(F2_BASE-100, F2_BASE+700, 300)
        self.line1, = self.ax1.plot(self.plot_x1, np.zeros(300), color='#39FF14', lw=2)
        self.line2, = self.ax2.plot(self.plot_x2, np.zeros(300), color='cyan', lw=2)
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#000')
            ax.set_ylim(0, 50)
            ax.axhline(y=THRESHOLD, color='red', ls='--', alpha=0.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack()

        self.root.protocol("WM_DELETE_WINDOW", self.emergency_exit)
        
        try:
            self.stream = sd.InputStream(callback=lambda *args: self.msg_queue.put(args[0].copy()), channels=1, samplerate=FS)
            self.stream.start()
        except: print("Mikrofon hatası!")

        self.update_loop()

    def clear_log(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete('1.0', tk.END)
        self.chat_area.config(state='disabled')
        self.is_block_active = False

    def emergency_exit(self):
        self.is_running = False
        os._exit(0)

    def add_to_display(self, text):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, text)
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

    def send_msg(self):
        msg = self.entry.get()
        if msg:
            ts = datetime.now().strftime("%H:%M:%S")
            self.add_to_display(f"\n[{ts}] SEN: {msg}\n")
            self.entry.delete(0, tk.END)
            threading.Thread(target=self.transmit, args=(msg,), daemon=True).start()

    def transmit(self, text):
        """Mesaj gönderimi ve sonrasında 1 saniye koruma beklemesi."""
        self.is_transmitting = True 
        time.sleep(0.1) 
        
        for char in text:
            val = ord(char)
            f1, f2 = F1_BASE + (val % 10) * STEP, F2_BASE + (val // 10) * STEP
            t = np.linspace(0, DURATION, int(FS * DURATION), False)
            win = windows.tukey(len(t), alpha=0.1)
            wave = ((np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)) / 2) * 1.0 * win
            sd.play(wave.astype(np.float32), FS)
            sd.wait() # Sesin bitmesini bekle
            time.sleep(0.05) # Karakterler arası kısa es
            
        # --- İSTEDİĞİN GÜNCELLEME: Gönderim bitince 1 saniye bekle ---
        print("[SİSTEM] Gönderim bitti, yankı koruması için 1 sn bekleniyor...")
        time.sleep(1.0) 
        
        self.is_transmitting = False 
        self.last_char_code = -1 
        print("[SİSTEM] Mikrofon tekrar aktif.")

    def update_loop(self):
        if not self.is_running: return

        # Gönderim veya koruma süresindeysek mikrofonu pas geç
        if self.is_transmitting:
            while not self.msg_queue.empty(): self.msg_queue.get_nowait()
            self.root.after(50, self.update_loop)
            return

        data = []
        while not self.msg_queue.empty(): data.append(self.msg_queue.get_nowait())
        
        if data:
            audio = np.concatenate(data).flatten()
            yf = fft(audio); xf = fftfreq(len(yf), 1/FS)
            mag = np.abs(yf[:len(yf)//2]); freqs = xf[:len(xf)//2]

            self.line1.set_ydata(np.interp(self.plot_x1, freqs, mag))
            self.line2.set_ydata(np.interp(self.plot_x2, freqs, mag))
            self.canvas.draw_idle()

            m1, m2 = (freqs >= F1_BASE-20) & (freqs <= F1_BASE+700), (freqs >= F2_BASE-20) & (freqs <= F2_BASE+700)
            
            v1, v2 = -1, -1
            if np.any(m1) and np.any(m2):
                idx1, idx2 = np.argmax(mag[m1]), np.argmax(mag[m2])
                if mag[m1][idx1] > THRESHOLD and mag[m2][idx2] > THRESHOLD:
                    v1, v2 = round((freqs[m1][idx1]-F1_BASE)/STEP), round((freqs[m2][idx2]-F2_BASE)/STEP)

            if v1 != -1 and v2 != -1:
                char_code = (v2 * 10) + v1
                if 32 <= char_code <= 126:
                    self.confidence_buffer.append(char_code)
                    if len(self.confidence_buffer) >= CONFIDENCE_REQ:
                        if all(x == char_code for x in self.confidence_buffer):
                            if char_code != self.last_char_code:
                                now = time.time()
                                if not self.is_block_active or (now - self.last_signal_time > LINE_TIMEOUT):
                                    ts = datetime.now().strftime("%H:%M:%S")
                                    self.add_to_display(f"\n[{ts}] GELEN: ")
                                    self.is_block_active = True
                                
                                self.add_to_display(chr(char_code))
                                self.last_char_code = char_code
                                self.last_signal_time = now
                        self.confidence_buffer.pop(0)
            else:
                if time.time() - self.last_signal_time > 0.7:
                    self.last_char_code = -1
                    self.confidence_buffer = []

        self.root.after(50, self.update_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SecureChatV14(root)
    root.mainloop()