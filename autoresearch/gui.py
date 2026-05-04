import os
import threading
import queue
import subprocess
from tkinter import filedialog

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

RESULTS_FILE = "results.tsv"


def detect_gpu():
    import torch

    if not torch.cuda.is_available():
        return None, 0, 0, (0, 0), False
    name = torch.cuda.get_device_name(0)
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    vram_gb = vram_mb / 1024
    cap = torch.cuda.get_device_capability()
    use_bf16 = cap >= (7, 5)
    return name, vram_gb, vram_mb, cap, use_bf16


class LitesearchApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Litesearch")
        self.geometry("960x640")
        self.minsize(700, 500)

        self.training_thread = None
        self.stop_event = None
        self.log_queue = None
        self.is_training = False
        self.current_config = None
        self.result = None
        self.experiment_count = 0
        self.export_after_n = None
        self.export_dir = None

        (
            self.gpu_name,
            self.gpu_vram_gb,
            self.gpu_vram_mb,
            self.gpu_cap,
            self.use_bf16,
        ) = detect_gpu()
        if self.gpu_name is None:
            self._show_no_gpu()
            return

        self.vram_var = ctk.DoubleVar(value=min(self.gpu_vram_gb, 32.0))
        self.lr_var = ctk.DoubleVar(value=0.04)

        self._build_ui()
        self._update_config()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _show_no_gpu(self):
        self.geometry("400x150")
        ctk.CTkLabel(
            self,
            text="No CUDA GPU found.\nLitesearch requires an NVIDIA GPU.",
            font=ctk.CTkFont(size=14),
            text_color="#ff6b6b",
        ).pack(expand=True)

    def _build_ui(self):
        PAD = {"padx": 16, "pady": 8}
        FONT = ctk.CTkFont(family="Consolas", size=12)
        FONT_SM = ctk.CTkFont(family="Consolas", size=11)
        FONT_LG = ctk.CTkFont(family="Consolas", size=13, weight="bold")
        FONT_TITLE = ctk.CTkFont(family="Consolas", size=20, weight="bold")

        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="Litesearch", font=FONT_TITLE, text_color="#4fc3f7"
        ).grid(row=0, column=0, sticky="w")
        dtype_str = "bf16" if self.use_bf16 else "fp32"
        ctk.CTkLabel(
            header,
            text=f"{self.gpu_name}  •  {self.gpu_vram_gb:.1f} GB  •  {dtype_str}",
            font=FONT_SM,
            text_color="#888888",
        ).grid(row=0, column=1, sticky="e")

        ctrl = ctk.CTkFrame(root)
        ctrl.grid(row=1, column=0, sticky="ew", **PAD)
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="VRAM", font=FONT).grid(
            row=0, column=0, padx=(12, 4), pady=6, sticky="w"
        )
        self.vram_slider = ctk.CTkSlider(
            ctrl,
            from_=1.0,
            to=32.0,
            variable=self.vram_var,
            number_of_steps=62,
            command=self._on_vram,
        )
        self.vram_slider.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.vram_lbl = ctk.CTkLabel(
            ctrl,
            text=f"{self.vram_var.get():.0f} GB",
            font=FONT_LG,
            text_color="#4fc3f7",
            width=60,
        )
        self.vram_lbl.grid(row=0, column=2, padx=(6, 12), pady=6)

        ctk.CTkLabel(ctrl, text="LR", font=FONT).grid(
            row=1, column=0, padx=(12, 4), pady=6, sticky="w"
        )
        self.lr_slider = ctk.CTkSlider(
            ctrl,
            from_=0.005,
            to=0.2,
            variable=self.lr_var,
            number_of_steps=40,
            command=self._on_lr,
        )
        self.lr_slider.grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        self.lr_lbl = ctk.CTkLabel(
            ctrl,
            text=f"{self.lr_var.get():.3f}",
            font=FONT_LG,
            text_color="#4fc3f7",
            width=60,
        )
        self.lr_lbl.grid(row=1, column=2, padx=(6, 12), pady=6)

        bot = ctk.CTkFrame(root)
        bot.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        bot.grid_columnconfigure(0, weight=1)

        self.config_lbl = ctk.CTkLabel(
            bot, text="", font=FONT_SM, text_color="#bbbbbb", anchor="w"
        )
        self.config_lbl.grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))

        btnrow = ctk.CTkFrame(bot, fg_color="transparent")
        btnrow.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.start_btn = ctk.CTkButton(
            btnrow,
            text="Start",
            font=FONT_LG,
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            height=34,
            width=120,
            command=self._on_start,
        )
        self.start_btn.pack(side="left")

        self.continue_btn = ctk.CTkButton(
            btnrow,
            text="Continue",
            font=FONT_SM,
            fg_color="#e65100",
            hover_color="#bf360c",
            height=34,
            width=90,
            command=self._on_continue,
            state="disabled",
        )
        self.continue_btn.pack(side="left", padx=(8, 0))

        self.stop_btn = ctk.CTkButton(
            btnrow,
            text="Stop",
            font=FONT_LG,
            fg_color="#c62828",
            hover_color="#b71c1c",
            height=34,
            width=80,
            command=self._on_stop,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.export_btn = ctk.CTkButton(
            btnrow,
            text="Export",
            font=FONT_SM,
            fg_color="#1565c0",
            hover_color="#0d47a1",
            height=34,
            width=80,
            command=self._on_export,
            state="disabled",
        )
        self.export_btn.pack(side="left", padx=(8, 0))

        self.schedule_btn = ctk.CTkButton(
            btnrow,
            text="Schedule",
            font=FONT_SM,
            fg_color="#4527a0",
            hover_color="#311b92",
            height=34,
            width=80,
            command=self._open_schedule_dialog,
        )
        self.schedule_btn.pack(side="left", padx=(8, 0))

        self.schedule_lbl = ctk.CTkLabel(
            btnrow, text="", font=FONT_SM, text_color="#888888"
        )
        self.schedule_lbl.pack(side="left", padx=(4, 0))

        self.try_btn = ctk.CTkButton(
            btnrow,
            text="Try",
            font=FONT_SM,
            fg_color="#00695c",
            hover_color="#004d40",
            height=34,
            width=60,
            command=self._open_try_dialog,
            state="disabled",
        )
        self.try_btn.pack(side="left", padx=(8, 0))

        self.status_lbl = ctk.CTkLabel(
            btnrow, text="Ready", font=FONT_SM, text_color="#888888"
        )
        self.status_lbl.pack(side="right")

        vbar = ctk.CTkFrame(bot, fg_color="transparent")
        vbar.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        vbar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(vbar, text="VRAM", font=FONT_SM, text_color="#888888").grid(
            row=0, column=0, padx=(0, 6)
        )
        self.vram_bar = ctk.CTkProgressBar(vbar, height=10)
        self.vram_bar.grid(row=0, column=1, sticky="ew")
        self.vram_bar.set(0)
        self.vram_txt = ctk.CTkLabel(
            vbar, text="", font=FONT_SM, text_color="#888888", width=100
        )
        self.vram_txt.grid(row=0, column=2, padx=(6, 0))

        term_frame = ctk.CTkFrame(root)
        term_frame.grid(row=3, column=0, sticky="nsew")
        term_frame.grid_columnconfigure(0, weight=1)
        term_frame.grid_rowconfigure(0, weight=1)

        self.terminal = ctk.CTkTextbox(
            term_frame,
            font=FONT_SM,
            text_color="#d4d4d4",
            fg_color="#0d0d1a",
            wrap="word",
        )
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._log("Ready. Adjust VRAM budget and click Start.\n\n")

    def _log(self, text):
        self.terminal.insert("end", text)
        self.terminal.see("end")
        lines = int(self.terminal.index("end-1c").split(".")[0])
        if lines > 600:
            self.terminal.delete("1.0", f"{lines - 500 + 1}.0")

    def _update_config(self):
        vram_mb = self.vram_var.get() * 1024
        from train import compute_optimal_config
        from prepare import VOCAB_SIZE

        try:
            cfg = compute_optimal_config(vram_mb, self.use_bf16, VOCAB_SIZE)
            self.current_config = cfg
            np = (
                cfg["depth"]
                * (4 * cfg["n_embd"] ** 2 + 2 * cfg["n_embd"] * 4 * cfg["n_embd"])
                + 2 * VOCAB_SIZE * cfg["n_embd"]
                + (cfg["depth"] // 2) * VOCAB_SIZE * cfg["n_embd"]
            )
            self.config_lbl.configure(
                text=f"depth={cfg['depth']}  d={cfg['n_embd']}  heads={cfg['n_head']}  "
                f"B={cfg['device_batch_size']}  T={cfg['max_seq_len']}  "
                f"~{np / 1e6:.0f}M params  ~{cfg['estimated_vram_mb']:.0f}MB"
            )
        except Exception as e:
            self.config_lbl.configure(text=str(e))
            self.current_config = None

    def _on_vram(self, val):
        self.vram_lbl.configure(text=f"{val:.0f} GB")
        if not self.is_training:
            self._update_config()

    def _on_lr(self, val):
        self.lr_lbl.configure(text=f"{val:.3f}")

    def _on_start(self):
        if self.is_training or self.current_config is None:
            return

        cache = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
        tok_dir = os.path.join(cache, "tokenizer")
        data_dir = os.path.join(cache, "data")
        if not os.path.exists(tok_dir) or not os.listdir(tok_dir):
            self._log("No tokenizer found. Run: python prepare.py\n")
            self.status_lbl.configure(text="Run prepare.py first", text_color="#ff6b6b")
            return
        if not os.path.exists(data_dir) or not any(
            f.endswith(".parquet") for f in os.listdir(data_dir)
        ):
            self._log("No data found. Run: python prepare.py\n")
            self.status_lbl.configure(text="Run prepare.py first", text_color="#ff6b6b")
            return

        self.is_training = True
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.result = None

        import torch

        torch.cuda.reset_peak_memory_stats()

        cfg = dict(self.current_config)
        lr = self.lr_var.get()

        self.terminal.delete("1.0", "end")
        self._log(
            f"Starting  •  VRAM budget {self.vram_var.get():.0f} GB  •  LR {lr:.3f}\n"
        )
        self._log(
            f"Config: depth={cfg['depth']} d={cfg['n_embd']} B={cfg['device_batch_size']} T={cfg['max_seq_len']}\n\n"
        )

        self.start_btn.configure(state="disabled")
        self.continue_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.export_btn.configure(state="disabled")
        self.try_btn.configure(state="disabled")
        self.vram_slider.configure(state="disabled")
        self.status_lbl.configure(text="Training...", text_color="#4fc3f7")

        self.training_thread = threading.Thread(
            target=self._worker, args=(cfg, lr), daemon=True
        )
        self.training_thread.start()
        self.after(100, self._poll)

    def _on_stop(self):
        if self.stop_event:
            self.stop_event.set()
        self.status_lbl.configure(text="Stopping...", text_color="#ffab40")

    def _on_continue(self):
        if self.is_training or not self.result or self.result.get("crashed"):
            return

        self.is_training = True
        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        prev_result = dict(self.result)
        lr = self.lr_var.get()

        import torch

        torch.cuda.reset_peak_memory_stats()

        self.terminal.delete("1.0", "end")
        self._log(f"Continuing  •  LR {lr:.3f}\n\n")

        self.start_btn.configure(state="disabled")
        self.continue_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.export_btn.configure(state="disabled")
        self.try_btn.configure(state="disabled")
        self.vram_slider.configure(state="disabled")
        self.status_lbl.configure(text="Training...", text_color="#4fc3f7")

        self.training_thread = threading.Thread(
            target=self._continue_worker, args=(prev_result, lr), daemon=True
        )
        self.training_thread.start()
        self.after(100, self._poll)

    def _continue_worker(self, prev_result, lr):
        try:
            from train import continue_training

            self.result = continue_training(
                prev_result,
                lr_override=lr,
                log_queue=self.log_queue,
                stop_event=self.stop_event,
            )
        except Exception as e:
            self.log_queue.put(f"\nCRASH: {e}\n")
            self.result = {"crashed": True, "val_bpb": 0.0, "peak_vram_mb": 0.0}
        self.log_queue.put("__DONE__")

    def _worker(self, cfg, lr):
        try:
            from train import run_training

            self.result = run_training(
                config=cfg,
                lr_override=lr,
                log_queue=self.log_queue,
                stop_event=self.stop_event,
            )
        except Exception as e:
            self.log_queue.put(f"\nCRASH: {e}\n")
            self.result = {"crashed": True, "val_bpb": 0.0, "peak_vram_mb": 0.0}
        self.log_queue.put("__DONE__")

    def _poll(self):
        if not self.is_training:
            return
        msgs = []
        try:
            while True:
                msgs.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        for m in msgs:
            if m == "__DONE__":
                self._on_done()
                return
            self._log(m)
        self._update_vram()
        self.after(100, self._poll)

    def _on_done(self):
        self.is_training = False
        self._update_vram()
        if self.result and not self.result.get("crashed"):
            self.experiment_count += 1
            self._save_tsv(self.result)
            self.export_btn.configure(state="normal")
            self.try_btn.configure(state="normal")
            self.continue_btn.configure(state="normal")
            self.status_lbl.configure(
                text=f"Done  •  val_bpb {self.result['val_bpb']:.6f}",
                text_color="#69f0ae",
            )
            if self.export_after_n and self.experiment_count >= self.export_after_n:
                self._do_export()
                self.export_after_n = None
                self.export_dir = None
                self.schedule_lbl.configure(text="")
        else:
            self.status_lbl.configure(text="Crashed", text_color="#ff6b6b")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.vram_slider.configure(state="normal")

    def _update_vram(self):
        import torch

        if not torch.cuda.is_available():
            return
        try:
            alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
            frac = min(alloc / self.gpu_vram_mb, 1.0) if self.gpu_vram_mb > 0 else 0
            self.vram_bar.set(frac)
            c = "#69f0ae" if frac < 0.5 else "#ffd54f" if frac < 0.8 else "#ff6b6b"
            self.vram_bar.configure(progress_color=c)
            self.vram_txt.configure(
                text=f"{alloc:.0f} / {self.gpu_vram_mb:.0f} MB", text_color=c
            )
        except Exception:
            pass

    def _save_tsv(self, r):
        try:
            try:
                commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--short", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                commit = "unknown"
            desc = f"depth={r.get('depth', '?')} d={r.get('n_embd', '?')} lr={self.lr_var.get():.3f}"
            status = "crash" if r.get("crashed") else "keep"
            if not os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, "w") as f:
                    f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")
            with open(RESULTS_FILE, "a") as f:
                f.write(
                    f"{commit}\t{r['val_bpb']:.6f}\t{r['peak_vram_mb'] / 1024:.1f}\t{status}\t{desc}\n"
                )
            self._log(f"Logged to {RESULTS_FILE}\n")
        except Exception as e:
            self._log(f"Log error: {e}\n")

    def _on_export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch model", "*.pth")],
            title="Export model",
        )
        if path:
            from train import export_model

            if export_model(self.result, path):
                self._log(f"Exported to {path}\n")

    def _do_export(self):
        from datetime import datetime
        from train import export_model

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.export_dir, f"litesearch_{ts}.pth")
        if export_model(self.result, path):
            self._log(f"Auto-exported to {path}\n")

    def _open_schedule_dialog(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Schedule Export")
        dlg.geometry("360x220")
        dlg.resizable(False, False)
        dlg.grab_set()

        FONT = ctk.CTkFont(family="Consolas", size=12)

        ctk.CTkLabel(dlg, text="Export after N experiments:", font=FONT).pack(
            padx=20, pady=(20, 4)
        )
        n_var = ctk.StringVar(value="5")
        entry = ctk.CTkEntry(dlg, textvariable=n_var, width=80, justify="center")
        entry.pack(pady=4)

        ctk.CTkLabel(dlg, text="Export directory:", font=FONT).pack(pady=(12, 4))
        dir_var = ctk.StringVar(value="exports")
        dir_entry = ctk.CTkEntry(dlg, textvariable=dir_var, width=280)
        dir_entry.pack(pady=4)

        def browse():
            d = filedialog.askdirectory(title="Choose export directory")
            if d:
                dir_var.set(d)

        ctk.CTkButton(
            dlg, text="Browse", font=ctk.CTkFont(size=11), width=70, command=browse
        ).pack(pady=2)

        def apply():
            try:
                n = int(n_var.get())
                d = dir_var.get()
                if n > 0 and d:
                    self.export_after_n = n
                    self.export_dir = d
                    os.makedirs(d, exist_ok=True)
                    self.schedule_lbl.configure(
                        text=f"Export after {n} runs", text_color="#4fc3f7"
                    )
                    self._log(f"Schedule: export after {n} experiments to {d}\n")
            except ValueError:
                pass
            dlg.destroy()

        btn_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_frame.pack(pady=(16, 0))
        ctk.CTkButton(btn_frame, text="Apply", width=80, command=apply).pack(
            side="left", padx=4
        )
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=80,
            fg_color="#555555",
            hover_color="#333333",
            command=dlg.destroy,
        ).pack(side="left", padx=4)

    def _open_try_dialog(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("Try model")
        dlg.geometry("500x460")
        dlg.resizable(False, False)
        dlg.grab_set()

        FONT = ctk.CTkFont(family="Consolas", size=12)
        FONT_SM = ctk.CTkFont(family="Consolas", size=11)

        page_settings = ctk.CTkFrame(dlg)
        page_settings.pack(fill="both", expand=True, padx=16, pady=16)

        page_output = ctk.CTkFrame(dlg)

        ctk.CTkLabel(page_settings, text="Prompt", font=FONT).pack(anchor="w")
        prompt_box = ctk.CTkTextbox(
            page_settings, height=120, font=FONT_SM, wrap="word"
        )
        prompt_box.pack(fill="x", pady=(4, 12))
        prompt_box.insert("1.0", "The meaning of life is")

        temp_var = ctk.DoubleVar(value=0.7)
        tokens_var = ctk.IntVar(value=128)
        top_p_var = ctk.DoubleVar(value=0.9)

        def slider_row(parent, label, var, from_, to, fmt):
            frame = ctk.CTkFrame(parent, fg_color="transparent")
            frame.pack(fill="x", pady=4)
            frame.grid_columnconfigure(1, weight=1)
            lbl = ctk.CTkLabel(frame, text=label, font=FONT_SM, width=100)
            lbl.grid(row=0, column=0, padx=(0, 6))
            ctk.CTkSlider(
                frame, from_=from_, to=to, variable=var, number_of_steps=40
            ).grid(row=0, column=1, sticky="ew")
            val_lbl = ctk.CTkLabel(frame, text=fmt(var.get()), font=FONT_SM, width=50)
            val_lbl.grid(row=0, column=2, padx=(6, 0))

            def update(*a):
                val_lbl.configure(text=fmt(var.get()))

            var.trace_add("write", update)
            return frame

        slider_row(
            page_settings, "Temperature", temp_var, 0.1, 2.0, lambda v: f"{v:.1f}"
        )
        slider_row(
            page_settings, "Max tokens", tokens_var, 16, 512, lambda v: f"{int(v)}"
        )
        slider_row(page_settings, "Top-p", top_p_var, 0.1, 1.0, lambda v: f"{v:.1f}")

        output_box = ctk.CTkTextbox(
            page_output, font=FONT_SM, wrap="word", fg_color="#0d0d1a"
        )
        output_box.pack(fill="both", expand=True, padx=16, pady=16)

        def do_generate():
            prompt = prompt_box.get("1.0", "end").strip()
            if not prompt:
                return
            page_settings.pack_forget()
            page_output.pack(fill="both", expand=True, padx=16, pady=16)
            output_box.delete("1.0", "end")
            output_box.insert("end", "Generating...\n")
            dlg.update()

            try:
                from train import generate

                tok = self._get_tokenizer()
                result = generate(
                    self.result["model"],
                    tok,
                    prompt,
                    max_tokens=tokens_var.get(),
                    temperature=temp_var.get(),
                    top_p=top_p_var.get(),
                )
                output_box.delete("1.0", "end")
                output_box.insert("end", result)
            except Exception as e:
                output_box.delete("1.0", "end")
                output_box.insert("end", f"Error: {e}")

        def do_back():
            if page_output.winfo_ismapped():
                page_output.pack_forget()
                page_settings.pack(fill="both", expand=True, padx=16, pady=16)
            else:
                dlg.destroy()

        btn_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_frame.pack(pady=(0, 12))

        gen_btn = ctk.CTkButton(
            btn_frame, text="Generate", width=100, command=do_generate
        )
        gen_btn.pack(side="left", padx=4)
        back_btn = ctk.CTkButton(
            btn_frame,
            text="Back",
            width=80,
            fg_color="#555555",
            hover_color="#333333",
            command=do_back,
        )
        back_btn.pack(side="left", padx=4)

    def _get_tokenizer(self):
        from prepare import Tokenizer

        return Tokenizer.from_directory()

    def _on_close(self):
        if self.is_training and self.stop_event:
            self.stop_event.set()
            if self.training_thread:
                self.training_thread.join(timeout=3)
        self.destroy()


if __name__ == "__main__":
    app = LitesearchApp()
    app.mainloop()
