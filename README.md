# Tái tạo chữ ký bằng đường cong B-spline

Ứng dụng Python cho phép đọc ảnh chữ ký, xử lý ảnh để trích xuất các điểm pixel thuộc nét chữ ký, sau đó tái tạo chữ ký bằng đường cong B-spline và xuất dữ liệu ra hai file `.dat` để mở bằng DUTMod/DISCO.

## 1. Chức năng chính

* Chọn ảnh chữ ký từ máy tính.
* Hiển thị ảnh chữ ký đã chọn trên giao diện.
* Xử lý ảnh và trích xuất các điểm pixel thuộc nét chữ ký.
* Xuất file `diempixel.dat` theo định dạng `[POINT]`.
* Tái tạo chữ ký bằng đường cong B-spline.
* Xuất file `bsplinecurve.dat` theo định dạng `[BSPLINECURVE]`.
* Hiển thị nội dung hai file kết quả trên giao diện.
* Hỗ trợ copy nội dung file kết quả.

## 2. Công nghệ sử dụng

* Python 3.12
* tkinter
* Pillow
* OpenCV
* NumPy

## 3. Cấu trúc thư mục

```txt
Signature_BSpline_App/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── output/
├── images/
└── src/
    ├── __init__.py
    ├── image_processing.py
    ├── dat_writer.py
    ├── bspline.py
    └── utils.py
```

Trong đó:

```txt
main.py
-> file chạy chính của chương trình, chứa giao diện tkinter.

src/image_processing.py
-> xử lý ảnh chữ ký, chuyển ảnh xám, nhị phân hoá và lấy điểm pixel.

src/dat_writer.py
-> ghi dữ liệu ra file diempixel.dat và bsplinecurve.dat.

src/bspline.py
-> cài đặt các hàm tính B-spline, basis function và least-square approximation.

src/utils.py
-> chứa các hàm hỗ trợ.

output/
-> chứa các file kết quả do chương trình sinh ra.

images/
-> chứa ảnh chữ ký dùng để thử nghiệm.
```

## 4. Yêu cầu trước khi chạy

Máy cần cài sẵn Python.

Khuyến nghị sử dụng:

```txt
Python 3.12.x
```

Kiểm tra Python:

```bash
python --version
```

Nếu kết quả hiển thị dạng sau là được:

```txt
Python 3.12.x
```

## 5. Cài đặt môi trường và thư viện

Mở terminal tại thư mục project.

Tạo môi trường ảo:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

Nếu dùng PowerShell và gặp lỗi:

```txt
running scripts is disabled on this system
```

thì chạy lệnh sau một lần:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Sau đó đóng terminal, mở lại terminal mới và kích hoạt lại môi trường ảo:

```bash
venv\Scripts\activate
```

Khi kích hoạt thành công, đầu dòng terminal sẽ xuất hiện `(venv)`.

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## 6. Chạy chương trình

Sau khi đã kích hoạt môi trường ảo và cài thư viện, chạy chương trình bằng lệnh:

```bash
python main.py
```

Giao diện chương trình sẽ hiển thị, cho phép chọn ảnh chữ ký và xử lý ảnh.

Ở những lần chạy sau, chỉ cần mở terminal tại thư mục project và chạy:

```bash
venv\Scripts\activate
python main.py
```

## 7. File kết quả

Sau khi xử lý thành công, chương trình sinh ra hai file trong thư mục `output`:

```txt
output/diempixel.dat
output/bsplinecurve.dat
```

Trong đó:

```txt
diempixel.dat
-> lưu các điểm pixel thuộc nét chữ ký theo định dạng [POINT] của DUTMod/DISCO.

bsplinecurve.dat
-> lưu các đường cong B-spline tái tạo theo định dạng [BSPLINECURVE] của DUTMod/DISCO.
```

Hai file này có thể được mở bằng DUTMod/DISCO để kiểm tra kết quả tái tạo chữ ký.