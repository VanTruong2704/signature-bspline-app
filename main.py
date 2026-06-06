import os
import traceback
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageGrab

from src.image_processing import extract_signature_strokes
from src.bspline import reconstruct_bspline_curves
from src.dat_writer import write_point_dat, write_bspline_dat
from src.utils import ensure_output_dir, read_text_file


APP_TITLE = "Tái tạo chữ ký bằng đường cong B-spline"

OUTPUT_DIR = "output"
POINT_FILE_NAME = "diempixel.dat"
BSPLINE_FILE_NAME = "bsplinecurve.dat"

SUPPORTED_IMAGE_TYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp"),
    ("PNG files", "*.png"),
    ("JPG files", "*.jpg *.jpeg"),
    ("BMP files", "*.bmp"),
    ("All files", "*.*"),
]


class SignatureBSplineApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1100x850")
        self.minsize(950, 720)

        self.current_image_path = None
        self.preview_image_tk = None
        self.is_processing = False

        self.point_file_path = os.path.join(OUTPUT_DIR, POINT_FILE_NAME)
        self.bspline_file_path = os.path.join(OUTPUT_DIR, BSPLINE_FILE_NAME)

        self._setup_style()
        self._build_ui()

    # =========================================================
    # STYLE
    # =========================================================
    def _setup_style(self):
        self.configure(bg="#f4f7fb")

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Main.TFrame",
            background="#f4f7fb"
        )

        style.configure(
            "Card.TFrame",
            background="#ffffff",
            relief="flat",
            borderwidth=1
        )

        style.configure(
            "Title.TLabel",
            background="#f4f7fb",
            foreground="#172033",
            font=("Segoe UI", 22, "bold")
        )

        style.configure(
            "Subtitle.TLabel",
            background="#f4f7fb",
            foreground="#5d6b82",
            font=("Segoe UI", 10)
        )

        style.configure(
            "CardTitle.TLabel",
            background="#ffffff",
            foreground="#172033",
            font=("Segoe UI", 12, "bold")
        )

        style.configure(
            "Normal.TLabel",
            background="#ffffff",
            foreground="#334155",
            font=("Segoe UI", 10)
        )

        style.configure(
            "Muted.TLabel",
            background="#ffffff",
            foreground="#64748b",
            font=("Segoe UI", 9)
        )

        style.configure(
            "Success.TLabel",
            background="#ffffff",
            foreground="#15803d",
            font=("Segoe UI", 10, "bold")
        )

        style.configure(
            "Error.TLabel",
            background="#ffffff",
            foreground="#dc2626",
            font=("Segoe UI", 10, "bold")
        )

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
            background="#2563eb",
            foreground="#ffffff"
        )

        style.map(
            "Primary.TButton",
            background=[
                ("active", "#1d4ed8"),
                ("pressed", "#1e40af")
            ],
            foreground=[
                ("active", "#ffffff"),
                ("pressed", "#ffffff")
            ]
        )

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
            background="#e2e8f0",
            foreground="#1e293b"
        )

        style.map(
            "Secondary.TButton",
            background=[
                ("active", "#cbd5e1"),
                ("pressed", "#94a3b8")
            ],
            foreground=[
                ("active", "#1e293b"),
                ("pressed", "#1e293b")
            ]
        )

        style.configure(
            "Copy.TButton",
            font=("Segoe UI", 9, "bold"),
            padding=(10, 5),
            background="#f1f5f9",
            foreground="#334155"
        )

        style.map(
            "Copy.TButton",
            background=[
                ("active", "#e2e8f0"),
                ("pressed", "#cbd5e1")
            ]
        )

    # =========================================================
    # UI
    # =========================================================
    def _build_ui(self):
        root = ttk.Frame(self, style="Main.TFrame", padding=22)
        root.pack(fill="both", expand=True)

        self._build_header(root)
        self._build_action_bar(root)

        content = ttk.Frame(root, style="Main.TFrame")
        content.pack(fill="both", expand=True, pady=(18, 0))

        content.columnconfigure(0, weight=4)
        content.columnconfigure(1, weight=6)
        content.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(content, style="Main.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        right_panel = ttk.Frame(content, style="Main.TFrame")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        left_panel.rowconfigure(0, weight=3)
        left_panel.rowconfigure(1, weight=2)
        left_panel.columnconfigure(0, weight=1)

        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        self._build_preview_card(left_panel)
        self._build_status_card(left_panel)

        self.point_text = self._build_file_card(
            parent=right_panel,
            row=0,
            title="diempixel.dat",
            copy_command=lambda: self._copy_text(self.point_text)
        )

        self.bspline_text = self._build_file_card(
            parent=right_panel,
            row=1,
            title="bsplinecurve.dat",
            copy_command=lambda: self._copy_text(self.bspline_text)
        )

    def _build_header(self, parent):
        header = ttk.Frame(parent, style="Main.TFrame")
        header.pack(fill="x")

        title = ttk.Label(
            header,
            text="TÁI TẠO CHỮ KÝ BẰNG ĐƯỜNG CONG B-SPLINE",
            style="Title.TLabel"
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header,
            text="Đọc ảnh chữ ký, trích xuất điểm pixel, tái tạo B-spline và xuất file .dat cho DUTMod/DISCO.",
            style="Subtitle.TLabel"
        )
        subtitle.pack(anchor="w", pady=(6, 0))

    def _build_action_bar(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.pack(fill="x", pady=(18, 0))

        card.columnconfigure(0, weight=0)
        card.columnconfigure(1, weight=0)
        card.columnconfigure(2, weight=1)

        upload_btn = ttk.Button(
            card,
            text="Upload ảnh chữ ký",
            style="Primary.TButton",
            command=self._choose_image
        )
        upload_btn.grid(row=0, column=0, sticky="w")

        paste_btn = ttk.Button(
            card,
            text="Dán ảnh từ clipboard",
            style="Secondary.TButton",
            command=self._paste_image_from_clipboard
        )
        paste_btn.grid(row=0, column=1, sticky="w", padx=(12, 0))

        hint = ttk.Label(
            card,
            text="Hỗ trợ: PNG, JPG, JPEG, BMP. Ảnh dán sẽ được lưu tạm trong thư mục output.",
            style="Muted.TLabel"
        )
        hint.grid(row=0, column=2, sticky="e", padx=(12, 0))

    def _build_preview_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))

        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        title = ttk.Label(card, text="Ảnh đã chọn", style="CardTitle.TLabel")
        title.grid(row=0, column=0, sticky="w")

        self.preview_canvas = tk.Canvas(
            card,
            bg="#f8fafc",
            highlightthickness=1,
            highlightbackground="#dbe3ef",
            relief="flat"
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        self.preview_canvas.bind("<Configure>", self._on_preview_resize)

        self._draw_preview_placeholder()

    def _build_status_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        card.columnconfigure(0, weight=1)

        title = ttk.Label(card, text="Trạng thái", style="CardTitle.TLabel")
        title.grid(row=0, column=0, sticky="w")

        self.status_label = ttk.Label(
            card,
            text="Chưa chọn ảnh chữ ký.",
            style="Normal.TLabel",
            wraplength=380,
            justify="left"
        )
        self.status_label.grid(row=1, column=0, sticky="nw", pady=(12, 0))

        self.path_label = ttk.Label(
            card,
            text="",
            style="Muted.TLabel",
            wraplength=380,
            justify="left"
        )
        self.path_label.grid(row=2, column=0, sticky="nw", pady=(12, 0))

    def _build_file_card(self, parent, row, title, copy_command):
        card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 12) if row == 0 else (12, 0))

        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        header = ttk.Frame(card, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")

        header.columnconfigure(0, weight=1)

        label = ttk.Label(header, text=title, style="CardTitle.TLabel")
        label.grid(row=0, column=0, sticky="w")

        copy_btn = ttk.Button(
            header,
            text="Copy",
            style="Copy.TButton",
            command=copy_command
        )
        copy_btn.grid(row=0, column=1, sticky="e")

        text_frame = ttk.Frame(card, style="Card.TFrame")
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        text_widget = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 9),
            bg="#f8fafc",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            padx=10,
            pady=10,
            undo=False
        )
        text_widget.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=text_widget.yview
        )
        y_scroll.grid(row=0, column=1, sticky="ns")

        x_scroll = ttk.Scrollbar(
            text_frame,
            orient="horizontal",
            command=text_widget.xview
        )
        x_scroll.grid(row=1, column=0, sticky="ew")

        text_widget.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )

        self._set_text_content(text_widget, "Nội dung file sẽ hiển thị ở đây sau khi xử lý ảnh.")

        return text_widget

    # =========================================================
    # IMAGE INPUT
    # =========================================================
    def _choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh chữ ký",
            filetypes=SUPPORTED_IMAGE_TYPES
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self._load_preview_image(file_path)
        self._process_current_image()

    def _paste_image_from_clipboard(self):
        try:
            clipboard_data = ImageGrab.grabclipboard()

            if clipboard_data is None:
                messagebox.showwarning(
                    "Không có ảnh",
                    "Clipboard hiện không có ảnh để dán."
                )
                return

            ensure_output_dir(OUTPUT_DIR)

            if isinstance(clipboard_data, Image.Image):
                image = clipboard_data.convert("RGB")
                temp_path = os.path.join(OUTPUT_DIR, "_clipboard_signature.png")
                image.save(temp_path)
                self.current_image_path = temp_path

            elif isinstance(clipboard_data, list):
                image_path = self._find_first_image_path(clipboard_data)

                if image_path is None:
                    messagebox.showwarning(
                        "Không có ảnh",
                        "Clipboard có dữ liệu, nhưng không tìm thấy file ảnh hợp lệ."
                    )
                    return

                self.current_image_path = image_path

            else:
                messagebox.showwarning(
                    "Không hỗ trợ",
                    "Dữ liệu trong clipboard không phải ảnh."
                )
                return

            self._load_preview_image(self.current_image_path)
            self._process_current_image()

        except Exception as e:
            self._show_error(
                "Lỗi khi dán ảnh từ clipboard.",
                e
            )

    def _find_first_image_path(self, file_paths):
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")

        for path in file_paths:
            if isinstance(path, str) and path.lower().endswith(valid_extensions):
                return path

        return None

    # =========================================================
    # PROCESSING
    # =========================================================
    def _process_current_image(self):
        if not self.current_image_path:
            return

        if self.is_processing:
            messagebox.showinfo(
                "Đang xử lý",
                "Chương trình đang xử lý ảnh hiện tại. Bạn hãy đợi xử lý xong rồi chọn ảnh khác."
            )
            return

        image_path = self.current_image_path
        self.is_processing = True
        self._set_status("Đang xử lý ảnh chữ ký...", status_type="normal")
        self.update_idletasks()

        worker = threading.Thread(
            target=self._process_current_image_worker,
            args=(image_path,),
            daemon=True
        )
        worker.start()

    def _process_current_image_worker(self, image_path):
        try:
            ensure_output_dir(OUTPUT_DIR)

            strokes = extract_signature_strokes(image_path)

            if not strokes:
                raise ValueError(
                    "Không trích xuất được điểm nào từ ảnh. "
                    "Bạn hãy thử ảnh chữ ký rõ hơn, nền sáng hơn hoặc nét chữ ký tối hơn."
                )

            write_point_dat(strokes, self.point_file_path)

            curves = reconstruct_bspline_curves(strokes)

            if not curves:
                raise ValueError(
                    "Không tái tạo được đường cong B-spline từ tập điểm đã trích xuất."
                )

            write_bspline_dat(curves, self.bspline_file_path)

            point_content = read_text_file(self.point_file_path)
            bspline_content = read_text_file(self.bspline_file_path)

            result = {
                "image_path": image_path,
                "point_content": point_content,
                "bspline_content": bspline_content,
                "total_points": sum(len(stroke) for stroke in strokes),
                "total_strokes": len(strokes),
                "total_curves": len(curves),
            }

            self.after(0, lambda: self._finish_processing_success(result))

        except Exception as e:
            self.after(0, lambda error=e: self._finish_processing_error(error))

    def _finish_processing_success(self, result):
        self.is_processing = False

        self._set_text_content(self.point_text, result["point_content"])
        self._set_text_content(self.bspline_text, result["bspline_content"])

        status_message = (
            "Đã tái tạo thành công.\n\n"
            f"Số nét/cụm điểm: {result['total_strokes']}\n"
            f"Tổng số điểm pixel: {result['total_points']}\n"
            f"Số đường cong B-spline: {result['total_curves']}\n\n"
            "Đã ghi file:\n"
            f"- {self.point_file_path}\n"
            f"- {self.bspline_file_path}"
        )

        self._set_status(status_message, status_type="success")

        self.path_label.configure(
            text=f"Ảnh đang xử lý:\n{result['image_path']}"
        )

    def _finish_processing_error(self, exception):
        self.is_processing = False
        self._show_error(
            "Lỗi trong quá trình xử lý ảnh chữ ký.",
            exception
        )

    # =========================================================
    # PREVIEW
    # =========================================================
    def _load_preview_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            self.original_preview_image = image
            self._draw_preview_image()

        except Exception as e:
            self._show_error(
                "Không thể hiển thị ảnh đã chọn.",
                e
            )

    def _on_preview_resize(self, event):
        if hasattr(self, "original_preview_image"):
            self._draw_preview_image()
        else:
            self._draw_preview_placeholder()

    def _draw_preview_placeholder(self):
        self.preview_canvas.delete("all")

        width = self.preview_canvas.winfo_width()
        height = self.preview_canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        self.preview_canvas.create_text(
            width // 2,
            height // 2,
            text="Ảnh chữ ký sẽ hiển thị ở đây",
            fill="#94a3b8",
            font=("Segoe UI", 12, "bold")
        )

    def _draw_preview_image(self):
        if not hasattr(self, "original_preview_image"):
            return

        self.preview_canvas.delete("all")

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        padding = 24
        max_width = max(1, canvas_width - padding * 2)
        max_height = max(1, canvas_height - padding * 2)

        image = self.original_preview_image.copy()
        image.thumbnail((max_width, max_height), Image.LANCZOS)

        self.preview_image_tk = ImageTk.PhotoImage(image)

        x = canvas_width // 2
        y = canvas_height // 2

        self.preview_canvas.create_image(
            x,
            y,
            image=self.preview_image_tk,
            anchor="center"
        )

    # =========================================================
    # TEXT / COPY / STATUS
    # =========================================================
    def _set_text_content(self, text_widget, content):
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("1.0", content)
        text_widget.configure(state="disabled")

    def _copy_text(self, text_widget):
        content = text_widget.get("1.0", "end-1c").strip()

        if not content:
            messagebox.showinfo("Copy", "Không có nội dung để copy.")
            return

        self.clipboard_clear()
        self.clipboard_append(content)
        self.update()

        messagebox.showinfo("Copy", "Đã copy nội dung vào clipboard.")

    def _set_status(self, message, status_type="normal"):
        if status_type == "success":
            style = "Success.TLabel"
        elif status_type == "error":
            style = "Error.TLabel"
        else:
            style = "Normal.TLabel"

        self.status_label.configure(
            text=message,
            style=style
        )

    def _show_error(self, user_message, exception):
        error_detail = str(exception)

        self._set_status(
            f"{user_message}\n\nChi tiết lỗi:\n{error_detail}",
            status_type="error"
        )

        print("========== ERROR ==========")
        print(user_message)
        print(error_detail)
        traceback.print_exc()
        print("===========================")


def main():
    app = SignatureBSplineApp()
    app.mainloop()


if __name__ == "__main__":
    main()