import numpy as np


# =========================================================
# CONFIG
# =========================================================

DEFAULT_DEGREE = 3
DEFAULT_KNOT_TYPE = 1

MIN_CONTROL_POINTS = 4
MAX_CONTROL_POINTS = 130

CONTROL_POINT_RATIO = 0.30


# =========================================================
# PUBLIC FUNCTION
# =========================================================

def reconstruct_bspline_curves(
    strokes,
    degree=DEFAULT_DEGREE,
    knot_type=DEFAULT_KNOT_TYPE,
    min_control_points=MIN_CONTROL_POINTS,
    max_control_points=MAX_CONTROL_POINTS,
    control_point_ratio=CONTROL_POINT_RATIO
):
    """
    Tái tạo danh sách đường cong B-spline từ các nét/cụm điểm.

    Tham số:
        strokes:
            Danh sách các nét/cụm điểm.

            Dạng:
                [
                    [(x1, y1), (x2, y2), ...],
                    [(x1, y1), (x2, y2), ...],
                    ...
                ]

        degree:
            Bậc của B-spline.
            Thường dùng degree = 3 để đường cong mượt.

        knot_type:
            UKnotType ghi ra file .dat.
            Theo file mẫu/test của DISCO, ta dùng knot_type = 1.

        min_control_points:
            Số điểm điều khiển tối thiểu.

        max_control_points:
            Số điểm điều khiển tối đa cho mỗi nét.

        control_point_ratio:
            Tỉ lệ chọn số điểm điều khiển theo số điểm dữ liệu.
            Ví dụ:
                200 điểm dữ liệu * 0.12 = khoảng 24 control points.

    Trả về:
        curves:
            [
                {
                    "degree": 3,
                    "knot_type": 1,
                    "control_points": [(x1, y1), (x2, y2), ...],
                    "knots": [u0, u1, u2, ...]
                },
                ...
            ]

    Ghi chú:
        Mỗi stroke sẽ tạo ra một block [BSPLINECURVE].
        Không ép toàn bộ chữ ký thành một đường cong duy nhất,
        vì chữ ký thường gồm nhiều nét/cụm khác nhau.
    """
    curves = []

    for stroke in strokes:
        if stroke is None or len(stroke) < 2:
            continue

        points = _to_numpy_points(stroke)

        if len(points) < 2:
            continue

        current_degree = min(int(degree), len(points) - 1)

        if current_degree < 1:
            continue

        num_control_points = _choose_num_control_points(
            num_data_points=len(points),
            degree=current_degree,
            min_control_points=min_control_points,
            max_control_points=max_control_points,
            ratio=control_point_ratio
        )

        if num_control_points <= current_degree:
            num_control_points = current_degree + 1

        if len(points) <= num_control_points:
            control_points = points.copy()
            knots = create_open_uniform_knot_vector(
                num_control_points=len(control_points),
                degree=current_degree
            )
        else:
            params = chord_length_parameterize(points)

            knots = create_averaging_knot_vector(
                params=params,
                num_control_points=num_control_points,
                degree=current_degree
            )

            control_points = least_square_approximation(
                data_points=points,
                params=params,
                knots=knots,
                degree=current_degree,
                num_control_points=num_control_points
            )

        curve = {
            "degree": current_degree,
            "knot_type": knot_type,
            "control_points": _numpy_points_to_list(control_points),
            "knots": [float(knot) for knot in knots]
        }

        curves.append(curve)

    return curves


# =========================================================
# PARAMETERIZATION
# =========================================================

def chord_length_parameterize(points):
    """
    Tạo tham số u cho tập điểm dữ liệu bằng chord-length parameterization.

    Ý tưởng:
        Khoảng cách giữa các điểm càng xa thì khoảng tham số càng lớn.
        Cách này thường tốt hơn chia đều tham số khi điểm dữ liệu không đều.

    Input:
        points: numpy array shape (m, 2)

    Output:
        params: numpy array shape (m,)
        params[0] = 0
        params[-1] = 1
    """
    n = len(points)

    if n == 1:
        return np.array([0.0], dtype=float)

    distances = np.zeros(n, dtype=float)

    for i in range(1, n):
        dx = points[i, 0] - points[i - 1, 0]
        dy = points[i, 1] - points[i - 1, 1]
        distances[i] = np.sqrt(dx * dx + dy * dy)

    total_length = np.sum(distances)

    if total_length <= 1e-12:
        return np.linspace(0.0, 1.0, n)

    params = np.cumsum(distances) / total_length

    params[0] = 0.0
    params[-1] = 1.0

    return params


# =========================================================
# KNOT VECTOR
# =========================================================

def create_open_uniform_knot_vector(num_control_points, degree):
    """
    Tạo knot vector mở đều.

    Dùng trong trường hợp số điểm dữ liệu ít,
    hoặc khi ta cần knot vector đơn giản.

    Với:
        n_control = số điểm điều khiển
        p = degree

    Số knot:
        n_control + p + 1

    Ví dụ:
        num_control_points = 7
        degree = 3

        Knot count = 7 + 3 + 1 = 11

        [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
    """
    num_control_points = int(num_control_points)
    degree = int(degree)

    knot_count = num_control_points + degree + 1
    knots = np.zeros(knot_count, dtype=float)

    for i in range(knot_count):
        if i <= degree:
            knots[i] = 0.0
        elif i >= num_control_points:
            knots[i] = 1.0
        else:
            denominator = num_control_points - degree
            knots[i] = (i - degree) / denominator

    return knots


def create_averaging_knot_vector(params, num_control_points, degree):
    """
    Tạo knot vector mở không đều bằng phương pháp averaging.

    Đây là cách phổ biến khi xấp xỉ B-spline bằng least-square.

    Input:
        params:
            Tham số u của các điểm dữ liệu, thường tạo bằng chord-length.

        num_control_points:
            Số điểm điều khiển cần tìm.

        degree:
            Bậc B-spline.

    Output:
        knots:
            Knot vector có độ dài:
                num_control_points + degree + 1

    Ghi chú:
        Knot đầu và cuối lặp degree + 1 lần.
        Các knot giữa lấy trung bình một nhóm tham số.
    """
    params = np.asarray(params, dtype=float)

    num_data_points = len(params)
    num_control_points = int(num_control_points)
    degree = int(degree)

    if num_control_points <= degree:
        raise ValueError("Số điểm điều khiển phải lớn hơn bậc B-spline.")

    if num_data_points <= num_control_points:
        return create_open_uniform_knot_vector(num_control_points, degree)

    knot_count = num_control_points + degree + 1
    knots = np.zeros(knot_count, dtype=float)

    knots[:degree + 1] = 0.0
    knots[num_control_points:] = 1.0

    number_of_internal_knots = num_control_points - degree - 1

    if number_of_internal_knots <= 0:
        return knots

    # Công thức averaging gần chuẩn:
    # knot[j + degree] = average(u[j] ... u[j + degree - 1])
    #
    # Tuy nhiên khi số data points nhiều hơn control points,
    # cần ánh xạ đều vị trí internal knot vào dãy tham số dữ liệu.
    for j in range(1, number_of_internal_knots + 1):
        start_float = j * (num_data_points - 1) / (number_of_internal_knots + 1)
        center = int(round(start_float))

        start = center - degree // 2
        end = start + degree

        start = max(1, start)
        end = min(num_data_points - 1, end)

        if end <= start:
            value = params[center]
        else:
            value = np.mean(params[start:end])

        knots[j + degree] = value

    knots = _fix_knot_vector_monotonic(knots)

    return knots


def _fix_knot_vector_monotonic(knots):
    """
    Đảm bảo knot vector không giảm.

    Do làm tròn chỉ số hoặc dữ liệu trùng nhau,
    đôi lúc internal knot có thể bằng hoặc nhỏ hơn knot trước đó.
    B-spline cho phép knot bằng nhau, nhưng để tránh lỗi chia 0 quá nhiều,
    ta ép tăng rất nhẹ ở vùng bên trong nếu cần.
    """
    knots = np.asarray(knots, dtype=float).copy()

    eps = 1e-8

    for i in range(1, len(knots) - 1):
        if knots[i] < knots[i - 1]:
            knots[i] = knots[i - 1]

        if 0.0 < knots[i] < 1.0 and knots[i] == knots[i - 1]:
            knots[i] = min(1.0, knots[i - 1] + eps)

    knots[0] = 0.0
    knots[-1] = 1.0

    return knots


# =========================================================
# BASIS FUNCTION
# =========================================================

def bspline_basis(i, degree, u, knots):
    """
    Tính basis function Ni,p(u) bằng công thức Cox-de Boor.

    Tham số:
        i:
            Chỉ số basis.

        degree:
            Bậc p.

        u:
            Giá trị tham số.

        knots:
            Knot vector.

    Công thức:
        Với p = 0:
            Ni,0(u) = 1 nếu knots[i] <= u < knots[i + 1]
            ngược lại = 0

        Với p > 0:
            Ni,p(u) =
                A * Ni,p-1(u) + B * Ni+1,p-1(u)

    Lưu ý:
        Tại u = 1, basis cuối cùng cần bằng 1
        để đường cong đi tới cuối miền tham số.
    """
    knots = np.asarray(knots, dtype=float)

    if degree == 0:
        if knots[i] <= u < knots[i + 1]:
            return 1.0

        if np.isclose(u, 1.0) and np.isclose(knots[i + 1], 1.0):
            return 1.0

        return 0.0

    left_denominator = knots[i + degree] - knots[i]
    right_denominator = knots[i + degree + 1] - knots[i + 1]

    left_value = 0.0
    right_value = 0.0

    if abs(left_denominator) > 1e-12:
        left_value = (
            (u - knots[i]) / left_denominator
        ) * bspline_basis(i, degree - 1, u, knots)

    if abs(right_denominator) > 1e-12:
        right_value = (
            (knots[i + degree + 1] - u) / right_denominator
        ) * bspline_basis(i + 1, degree - 1, u, knots)

    return left_value + right_value


def build_basis_matrix(params, knots, degree, num_control_points):
    """
    Tạo ma trận cơ sở B-spline.

    Bản tối ưu: không gọi đệ quy bspline_basis cho từng ô nữa.
    Mỗi giá trị u được tính theo quy hoạch động Cox-de Boor,
    giúp giảm CPU rõ rệt khi số điểm dữ liệu/control point tăng.
    """
    params = np.asarray(params, dtype=float)
    knots = np.asarray(knots, dtype=float)

    matrix = np.zeros((len(params), num_control_points), dtype=float)

    for row, u in enumerate(params):
        matrix[row, :] = _bspline_basis_row_fast(
            u=float(u),
            knots=knots,
            degree=int(degree),
            num_control_points=int(num_control_points)
        )

    return matrix


def _bspline_basis_row_fast(u, knots, degree, num_control_points):
    """
    Tính toàn bộ basis Ni,p(u) cho một giá trị u bằng quy hoạch động.
    Hàm này cho kết quả tương đương Cox-de Boor nhưng nhanh hơn bản đệ quy.
    """
    result = np.zeros(num_control_points, dtype=float)

    if num_control_points <= 0:
        return result

    if u <= knots[0]:
        result[0] = 1.0
        return result

    if u >= knots[-1] or np.isclose(u, knots[-1]):
        result[-1] = 1.0
        return result

    # Cần thêm degree basis phụ trong quá trình nâng bậc.
    work_size = num_control_points + degree
    basis = np.zeros(work_size, dtype=float)

    for i in range(work_size):
        if knots[i] <= u < knots[i + 1]:
            basis[i] = 1.0

    for k in range(1, degree + 1):
        next_basis = np.zeros(work_size, dtype=float)

        for i in range(work_size - k):
            left_denominator = knots[i + k] - knots[i]
            right_denominator = knots[i + k + 1] - knots[i + 1]

            left_value = 0.0
            right_value = 0.0

            if abs(left_denominator) > 1e-12:
                left_value = ((u - knots[i]) / left_denominator) * basis[i]

            if abs(right_denominator) > 1e-12:
                right_value = ((knots[i + k + 1] - u) / right_denominator) * basis[i + 1]

            next_basis[i] = left_value + right_value

        basis = next_basis

    result[:] = basis[:num_control_points]
    return result


# =========================================================
# LEAST SQUARE
# =========================================================

def least_square_approximation(
    data_points,
    params,
    knots,
    degree,
    num_control_points
):
    """
    Giải bài toán least-square để tìm điểm điều khiển B-spline.

    Input:
        data_points:
            Qk, numpy array shape (m, 2)

        params:
            uk, numpy array shape (m,)

        knots:
            knot vector

        degree:
            bậc B-spline

        num_control_points:
            số điểm điều khiển cần tìm

    Bài toán:
        C(u) = Σ Ni,p(u) * Pi

        Tìm Pi sao cho:
            tổng |Qk - C(uk)|^2 nhỏ nhất

    Dạng ma trận:
        A * P ≈ Q

        A: basis matrix
        P: control points cần tìm
        Q: data points

    Giải bằng:
        numpy.linalg.lstsq(A, Q)
    """
    data_points = np.asarray(data_points, dtype=float)
    params = np.asarray(params, dtype=float)
    knots = np.asarray(knots, dtype=float)

    basis_matrix = build_basis_matrix(
        params=params,
        knots=knots,
        degree=degree,
        num_control_points=num_control_points
    )

    # Ép đường cong bám điểm đầu và cuối tốt hơn:
    # thêm 2 dòng có trọng số lớn để điểm đầu/cuối gần Q0/Qm.
    weighted_A, weighted_Q = _add_endpoint_constraints(
        basis_matrix,
        data_points,
        weight=20.0
    )

    try:
        control_points, _, _, _ = np.linalg.lstsq(
            weighted_A,
            weighted_Q,
            rcond=None
        )
    except np.linalg.LinAlgError:
        # Nếu least-square lỗi do ma trận xấu,
        # dùng cách fallback: lấy mẫu trực tiếp từ data points.
        control_points = _fallback_control_points(
            data_points,
            num_control_points
        )

    control_points[0] = data_points[0]
    control_points[-1] = data_points[-1]

    return control_points


def _add_endpoint_constraints(basis_matrix, data_points, weight=20.0):
    """
    Thêm ràng buộc mềm cho điểm đầu và điểm cuối.

    Vì least-square chỉ xấp xỉ nên đôi khi đầu/cuối bị lệch.
    Ta thêm 2 phương trình có trọng số lớn:

        P0 gần Q0
        Pn gần Qm

    Điều này giúp đường cong bắt đầu/kết thúc hợp lý hơn.
    """
    num_control_points = basis_matrix.shape[1]

    start_row = np.zeros((1, num_control_points), dtype=float)
    end_row = np.zeros((1, num_control_points), dtype=float)

    start_row[0, 0] = weight
    end_row[0, -1] = weight

    start_point = data_points[0:1] * weight
    end_point = data_points[-1:] * weight

    weighted_A = np.vstack([
        basis_matrix,
        start_row,
        end_row
    ])

    weighted_Q = np.vstack([
        data_points,
        start_point,
        end_point
    ])

    return weighted_A, weighted_Q


def _fallback_control_points(data_points, num_control_points):
    """
    Cách dự phòng nếu least-square không giải được.

    Lấy đều các điểm dữ liệu làm điểm điều khiển.
    """
    indices = np.linspace(
        0,
        len(data_points) - 1,
        num_control_points,
        dtype=int
    )

    return data_points[indices].copy()


# =========================================================
# CURVE EVALUATION
# =========================================================

def evaluate_bspline_curve(control_points, knots, degree, sample_count=200):
    """
    Tính các điểm nằm trên đường cong B-spline.

    Hàm này không bắt buộc để ghi file .dat,
    nhưng hữu ích nếu sau này muốn vẽ preview đường cong trên giao diện.

    Trả về:
        curve_points: numpy array shape (sample_count, 2)
    """
    control_points = np.asarray(control_points, dtype=float)
    knots = np.asarray(knots, dtype=float)

    num_control_points = len(control_points)

    us = np.linspace(0.0, 1.0, sample_count)
    curve_points = []

    for u in us:
        point = np.zeros(2, dtype=float)

        for i in range(num_control_points):
            basis_value = bspline_basis(i, degree, u, knots)
            point += basis_value * control_points[i]

        curve_points.append(point)

    return np.asarray(curve_points, dtype=float)


# =========================================================
# CONTROL POINT CHOICE
# =========================================================

def _choose_num_control_points(
    num_data_points,
    degree,
    min_control_points,
    max_control_points,
    ratio
):
    """
    Chọn số điểm điều khiển cho một nét.

    Nguyên tắc:
        - Phải lớn hơn degree.
        - Không quá ít, nếu không đường cong lệch nhiều.
        - Không quá nhiều, nếu không đường cong bị rối và file nặng.
        - Tỉ lệ theo số điểm dữ liệu.

    Ví dụ:
        100 điểm dữ liệu -> khoảng 12 control points.
        300 điểm dữ liệu -> khoảng 36 control points.
    """
    num_data_points = int(num_data_points)
    degree = int(degree)

    estimated = int(round(num_data_points * ratio))

    lower_bound = max(degree + 1, min_control_points)
    upper_bound = min(max_control_points, num_data_points)

    if upper_bound < lower_bound:
        return upper_bound

    return max(lower_bound, min(estimated, upper_bound))


# =========================================================
# DATA HELPERS
# =========================================================

def _to_numpy_points(points):
    """
    Chuyển list điểm sang numpy array shape (n, 2).

    Hỗ trợ:
        (x, y)
        [x, y]
        {"x": x, "y": y}
        object có .x và .y
    """
    result = []

    for point in points:
        if isinstance(point, dict):
            x = point["x"]
            y = point["y"]

        elif hasattr(point, "x") and hasattr(point, "y"):
            x = point.x
            y = point.y

        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x = point[0]
            y = point[1]

        else:
            raise ValueError(f"Điểm không hợp lệ: {point}")

        result.append([float(x), float(y)])

    return np.asarray(result, dtype=float)


def _numpy_points_to_list(points):
    """
    Chuyển numpy array shape (n, 2) sang list tuple.

    Output:
        [(x1, y1), (x2, y2), ...]
    """
    result = []

    for point in points:
        result.append((float(point[0]), float(point[1])))

    return result