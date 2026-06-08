import numpy as np


DEFAULT_DEGREE = 3
DEFAULT_KNOT_TYPE = 1
MIN_CONTROL_POINTS = 4
MAX_CONTROL_POINTS = 130
CONTROL_POINT_RATIO = 0.30


def reconstruct_bspline_curves(
    strokes,
    degree=DEFAULT_DEGREE,
    knot_type=DEFAULT_KNOT_TYPE,
    min_control_points=MIN_CONTROL_POINTS,
    max_control_points=MAX_CONTROL_POINTS,
    control_point_ratio=CONTROL_POINT_RATIO
):
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
        curves.append({
            "degree": current_degree,
            "knot_type": knot_type,
            "control_points": _numpy_points_to_list(control_points),
            "knots": [float(knot) for knot in knots]
        })
    return curves


def chord_length_parameterize(points):
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


def create_open_uniform_knot_vector(num_control_points, degree):
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
    return _fix_knot_vector_monotonic(knots)


def _fix_knot_vector_monotonic(knots):
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


def build_basis_matrix(params, knots, degree, num_control_points):
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
    result = np.zeros(num_control_points, dtype=float)
    if num_control_points <= 0:
        return result
    if u <= knots[0]:
        result[0] = 1.0
        return result
    if u >= knots[-1] or np.isclose(u, knots[-1]):
        result[-1] = 1.0
        return result
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


def least_square_approximation(
    data_points,
    params,
    knots,
    degree,
    num_control_points
):
    data_points = np.asarray(data_points, dtype=float)
    params = np.asarray(params, dtype=float)
    knots = np.asarray(knots, dtype=float)
    basis_matrix = build_basis_matrix(
        params=params,
        knots=knots,
        degree=degree,
        num_control_points=num_control_points
    )
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
        control_points = _fallback_control_points(
            data_points,
            num_control_points
        )
    control_points[0] = data_points[0]
    control_points[-1] = data_points[-1]
    return control_points


def _add_endpoint_constraints(basis_matrix, data_points, weight=20.0):
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
    indices = np.linspace(
        0,
        len(data_points) - 1,
        num_control_points,
        dtype=int
    )
    return data_points[indices].copy()


def _choose_num_control_points(
    num_data_points,
    degree,
    min_control_points,
    max_control_points,
    ratio
):
    num_data_points = int(num_data_points)
    degree = int(degree)
    estimated = int(round(num_data_points * ratio))
    lower_bound = max(degree + 1, min_control_points)
    upper_bound = min(max_control_points, num_data_points)
    if upper_bound < lower_bound:
        return upper_bound
    return max(lower_bound, min(estimated, upper_bound))


def _to_numpy_points(points):
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
    return [(float(point[0]), float(point[1])) for point in points]
