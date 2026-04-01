import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from scipy.signal import windows
import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import time
from datetime import datetime
import os
import zlib

# --- QUAD-CHANNEL PARAMETRELERİ ---
FS = 44100
WAKE_UP_FREQ = 9500 
# 4 Kanal Frekans Havuzları
CHANNELS = [
    (10000, 11000), # Kanal 1
    (12000, 13000), # Kanal 2
    (14000, 15000), # Kanal 3
    (16000, 17000)  # Kanal 4
]
STEP = 60           
THRESHOLD = 2.5     
DEF_DURATION = 0.5    
DEF_GAP = 0.05         
COOLDOWN = 1.0       

class QuadFrameSyncedNodeV33:
    def __init__(self, root):
        self.root = root
        self.root.title("Acoustic Messenger v33 - Quad Frame Sync")
        self.root.geometry("1400x950")
        self.root.configure(bg="#f8f9fa")

        # --- DURUM DEĞİŞKENLERİ ---
        self.is_running = True
        self.duration = DEF_DURATION
        self.gap = DEF_GAP
        self.msg_queue = queue.Queue()
        self.is_transmitting = False
        self.is_listening = False
        self.is_channel_busy = False
        self.current_packet = []
        self.last_signal_time = 0
        self.last_frame_set = set() # Tekrar okumayı önleyen kilit

        self.apply_styles()
        self.setup_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.emergency_exit)
        try:
            self.stream = sd.InputStream(callback=lambda *args: self.msg_queue.put(args[0].copy()), 
                                         channels=1, samplerate=FS, blocksize=2048)
            self.stream.start()
        except: print("Mikrofon hatası!")

        self.update_loop()

    def apply_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Elegant.Horizontal.TProgressbar", background='#0078d4', thickness=22)

    def setup_ui(self):
        # --- SIDEBAR ---
        self.sidebar = tk.Frame(self.root, bg="#ffffff", width=400, padx=25, pady=25, highlightthickness=1, highlightbackground="#e1e1e1")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="KONTROL PANELİ", bg="#ffffff", font=("Segoe UI", 16, "bold")).pack(pady=(0, 20))
        
        tk.Label(self.sidebar, text="Kendi ID (Local):", bg="#ffffff", font=("Segoe UI", 12)).pack(anchor="w")
        self.my_id_ent = tk.Entry(self.sidebar, bg="#f3f3f3", font=("Segoe UI", 13), justify='center', borderwidth=0)
        self.my_id_ent.insert(0, "1"); self.my_id_ent.pack(fill=tk.X, pady=(5, 15))

        tk.Label(self.sidebar, text="Hedef ID (Remote):", bg="#ffffff", font=("Segoe UI", 12)).pack(anchor="w")
        self.target_id_ent = tk.Entry(self.sidebar, bg="#f3f3f3", font=("Segoe UI", 13), justify='center', borderwidth=0)
        self.target_id_ent.insert(0, "0"); self.target_id_ent.pack(fill=tk.X, pady=(5, 15))

        tk.Label(self.sidebar, text="HIZ AYARLARI", bg="#ffffff", fg="#0078d4", font=("Segoe UI", 12, "bold")).pack(pady=10)
        self.dur_scale = tk.Scale(self.sidebar, from_=0.05, to=1.0, label="Karakter Süresi (sn)", resolution=0.01, orient=tk.HORIZONTAL, bg="#ffffff", command=self.update_params)
        self.dur_scale.set(DEF_DURATION); self.dur_scale.pack(fill=tk.X)

        self.gap_scale = tk.Scale(self.sidebar, from_=0.01, to=0.5, label="Boşluk Süresi (sn)", resolution=0.01, orient=tk.HORIZONTAL, bg="#ffffff", command=self.update_params)
        self.gap_scale.set(DEF_GAP); self.gap_scale.pack(fill=tk.X, pady=10)

        tk.Button(self.sidebar, text="Varsayılanı Geri Yükle", fg="#d9534f", font=("Segoe UI", 10, "bold"), relief="flat", command=self.reset_defaults).pack(fill=tk.X)

        # Spektrum Grafiği
        tk.Label(self.sidebar, text="SPEKTRUM ANALİZİ (kHz)", bg="#ffffff", font=("Segoe UI", 10, "bold")).pack(pady=(20, 5))
        self.fig, self.ax = plt.subplots(figsize=(4, 3.5)); self.fig.patch.set_facecolor('#ffffff')
        self.plot_x = np.linspace(9000, 18000, 600)
        self.line, = self.ax.plot(self.plot_x, np.zeros(600), color='#0078d4', lw=2)
        self.ax.set_facecolor('#f8f9fa'); self.ax.set_ylim(0, 60); self.ax.set_xlim(9000, 18000)
        self.ax.set_xticks([9000, 11000, 13000, 15000, 17000])
        self.ax.set_xticklabels(['9k', '11k', '13k', '15k', '17k'])
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.sidebar); self.canvas.get_tk_widget().pack(fill=tk.X)

        # --- SAĞ PANEL ---
        self.main_panel = tk.Frame(self.root, bg="#f8f9fa", padx=25, pady=25)
        self.main_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.status_lbl = tk.Label(self.main_panel, text="● Kanal Müsait", bg="#f8f9fa", fg="#28a745", font=("Segoe UI", 12, "bold"))
        self.status_lbl.pack(anchor="e")

        self.chat_area = scrolledtext.ScrolledText(self.main_panel, bg="#ffffff", font=("Segoe UI", 14), padx=20, pady=20, borderwidth=0)
        self.chat_area.pack(fill=tk.BOTH, expand=True, pady=10)

        # Progress ve Sayaç
        prog_frame = tk.Frame(self.main_panel, bg="#f8f9fa")
        prog_frame.pack(fill=tk.X, pady=5)
        self.progress = ttk.Progressbar(prog_frame, orient=tk.HORIZONTAL, mode='determinate', style="Elegant.Horizontal.TProgressbar")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.timer_lbl = tk.Label(prog_frame, text="Kalan Süre: 0s", bg="#f8f9fa", fg="#0078d4", font=("Segoe UI", 12, "bold"))
        self.timer_lbl.pack(side=tk.RIGHT, padx=15)

        # Giriş
        entry_container = tk.Frame(self.main_panel, bg="#ffffff", padx=15, pady=15, highlightthickness=1, highlightbackground="#ced4da")
        entry_container.pack(fill=tk.X, pady=10)
        self.entry = tk.Entry(entry_container, bg="#ffffff", font=("Segoe UI", 15), borderwidth=0)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.send_action())
        
        tk.Button(entry_container, text="Gönder", bg="#0078d4", fg="#fff", font=("Segoe UI", 12, "bold"), relief="flat", padx=35, pady=10, command=self.send_action).pack(side=tk.LEFT, padx=10)
        tk.Button(entry_container, text="Temizle", font=("Segoe UI", 12), relief="flat", command=self.clear_log).pack(side=tk.LEFT)

    def update_params(self, _=None):
        self.duration = self.dur_scale.get(); self.gap = self.gap_scale.get()

    def reset_defaults(self):
        self.dur_scale.set(DEF_DURATION); self.gap_scale.set(DEF_GAP); self.update_params()

    def clear_log(self):
        self.chat_area.config(state='normal'); self.chat_area.delete('1.0', tk.END); self.chat_area.config(state='disabled')

    def emergency_exit(self):
        self.is_running = False; os._exit(0)

    def add_text(self, text, is_own=False):
        self.chat_area.config(state='normal'); tag = "own" if is_own else "incoming"
        self.chat_area.insert(tk.END, text, tag)
        self.chat_area.tag_config("own", foreground="#0078d4", font=("Segoe UI", 14, "bold"))
        self.chat_area.tag_config("incoming", foreground="#2c3e50")
        self.chat_area.see(tk.END); self.chat_area.config(state='disabled')

    def send_action(self):
        msg = self.entry.get()
        if msg:
            my_id, tgt_id = int(self.my_id_ent.get()), int(self.target_id_ent.get())
            self.entry.delete(0, tk.END)
            threading.Thread(target=self.transmit, args=(my_id, tgt_id, msg), daemon=True).start()

    def transmit(self, src, dst, text):
        while self.is_channel_busy: time.sleep(0.1)
        self.is_transmitting = True
        
        payload = text.encode('utf-8')
        raw = [src, dst, len(payload), zlib.crc32(payload) & 0xFF] + list(payload)
        while len(raw) % 4 != 0: raw.append(0) # Dolgu (Null padding)

        total_est = 0.5 + ((len(raw)//4) * (self.duration + self.gap))
        start_t = time.time(); ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # 1. Wake-up
        sd.play(np.sin(2*np.pi*WAKE_UP_FREQ*np.linspace(0,0.4,int(FS*0.4)))*windows.tukey(int(FS*0.4)), FS); sd.wait(); time.sleep(0.1)

        # 2. Quad-Channel Data
        for i in range(0, len(raw), 4):
            elapsed = time.time() - start_t
            self.timer_lbl.config(text=f"Kalan Süre: {max(0, int(total_est - elapsed))}s")
            self.progress['value'] = (elapsed / total_est) * 100
            
            chars = raw[i:i+4]
            t = np.linspace(0, self.duration, int(FS*self.duration))
            combined_wave = np.zeros_like(t)
            
            for ch_idx, char_val in enumerate(chars):
                base1, base2 = CHANNELS[ch_idx]
                f1 = base1 + (char_val % 10) * STEP
                f2 = base2 + (char_val // 10) * STEP
                combined_wave += (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t))
            
            final_wave = (combined_wave / 8) * windows.tukey(len(t))
            sd.play(final_wave.astype(np.float32), FS); sd.wait(); time.sleep(self.gap)

        self.progress['value'] = 100
        self.add_text(f"\n[{ts}] ÇIKIŞ -> Hedef:{dst} | Boyut: {len(payload)} byte | Süre: {time.time()-start_t:.2f} sn\nSİZ: {text}\n", True)
        time.sleep(0.5); self.progress['value'] = 0; self.is_transmitting = False

    def update_loop(self):
        if not self.is_running: return
        data = []
        while not self.msg_queue.empty(): data.append(self.msg_queue.get_nowait())
        
        if data and not self.is_transmitting:
            audio = np.concatenate(data).flatten()
            yf = fft(audio); xf = fftfreq(len(yf), 1/FS)
            mag = np.abs(yf[:len(yf)//2]); freqs = xf[:len(xf)//2]
            self.line.set_ydata(np.interp(self.plot_x, freqs, mag)); self.canvas.draw_idle()

            peak = np.max(mag[(freqs > 9000) & (freqs < 18000)])
            if peak > THRESHOLD:
                self.is_channel_busy = True; self.status_lbl.config(text="● Kanal Meşgul", fg="#d9534f")
            else:
                self.is_channel_busy = False; self.status_lbl.config(text="● Kanal Müsait", fg="#28a745")

            if not self.is_listening and np.max(mag[(freqs > WAKE_UP_FREQ-50) & (freqs < WAKE_UP_FREQ+50)]) > THRESHOLD:
                self.is_listening = True; self.start_t = time.time()
                self.arrival_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            if self.is_listening:
                frame_chars = []
                for b1, b2 in CHANNELS:
                    m1, m2 = (freqs >= b1-20) & (freqs <= b1+700), (freqs >= b2-20) & (freqs <= b2+700)
                    if np.any(m1) and np.any(m2):
                        i1, i2 = np.argmax(mag[m1]), np.argmax(mag[m2])
                        if mag[m1][i1] > THRESHOLD and mag[m2][i2] > THRESHOLD:
                            char_code = (round((freqs[m2][i2]-b2)/STEP)*10)+round((freqs[m1][i1]-b1)/STEP)
                            frame_chars.append(char_code)

                # SYNC LOGIC: Sadece yeni bir "Karakter Seti" duyduğumuzda ekle
                if len(frame_chars) == 4:
                    current_set = tuple(frame_chars)
                    if current_set != self.last_frame_set:
                        self.current_packet.extend(frame_chars)
                        self.last_frame_set = current_set
                        self.last_signal_time = time.time()
                
                if time.time() - self.last_signal_time > 1.5 and self.current_packet:
                    self.finalize()

        self.root.after(50, self.update_loop)

    def finalize(self):
        total_dur = time.time() - self.start_t - 1.5
        if len(self.current_packet) >= 4:
            src, dst, sz, crc = self.current_packet[:4]
            # Paketi temizle ve decode et
            msg_data = bytes([c for c in self.current_packet[4:] if c != 0])[:sz]
            msg = msg_data.decode('utf-8', errors='replace')
            if dst == int(self.my_id_ent.get()) or dst == 0:
                status = "TAM" if (zlib.crc32(msg_data) & 0xFF) == crc else "HATA"
                self.add_text(f"\n[{self.arrival_ts}] GİRİŞ <- Kimden:{src} | Boyut: {sz} byte | Süre: {total_dur:.2f} sn | {status}\nGELEN: {msg}\n")
        
        self.current_packet = []; self.is_listening = False; self.last_frame_set = None

if __name__ == "__main__":
    root = tk.Tk(); app = QuadFrameSyncedNodeV33(root); root.mainloop()