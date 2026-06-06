import tkinter as tk


def main():
    root = tk.Tk()
    root.title("Tái tạo chữ ký bằng đường cong B-spline")
    root.geometry("900x700")

    title = tk.Label(
        root,
        text="TÁI TẠO CHỮ KÝ BẰNG ĐƯỜNG CONG B-SPLINE",
        font=("Arial", 16, "bold")
    )
    title.pack(pady=20)

    btn_upload = tk.Button(
        root,
        text="Upload ảnh chữ ký",
        font=("Arial", 12)
    )
    btn_upload.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()