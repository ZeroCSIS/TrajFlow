import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from scipy.interpolate import interp1d, make_interp_spline, LSQUnivariateSpline
from scipy.linalg import LinAlgError
from rdp import rdp  # <<< Import RDP


# --- Functions: compute_arc_length_t, detect_anchors, _calculate_knots, _calculate_curvature (keep as before) ---
# (Implementations from previous responses)
# ... (ensure these helper functions are present) ...
def compute_arc_length_t(points):  # ... (implementation) ...
    """Computes the arc length parameterization and returns uniformly sampled points."""
    points = np.asarray(points);
    N = len(points)
    if N < 2:
        # If less than 2 points, return linear space and repeat the point(s)
        t_uniform = np.linspace(0, 1, N, endpoint=True) if N > 0 else np.array([])
        points_uniform = np.repeat(points, N, axis=0) if N > 0 else np.empty((0, points.shape[1]))
        return t_uniform, points_uniform

    # Calculate segment lengths and cumulative length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    s = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = s[-1]

    # Normalize cumulative lengths to get original parameterization t_orig
    if total_length < 1e-9:
        # Handle case where all points are coincident or very close
        t_orig = np.linspace(0, 1, N, endpoint=True)
    else:
        t_orig = s / total_length

    # Ensure t_orig starts at 0 and ends at 1
    t_orig[0] = 0.0
    t_orig[-1] = 1.0

    # Create target uniform parameterization t_uniform
    t_uniform = np.linspace(0, 1, N, endpoint=True)

    # Remove duplicate points based on t_orig to avoid interpolation issues
    # Keep first and last points always
    unique_mask = np.ones(N, dtype=bool)
    if N > 2:
        unique_mask[1:-1] = np.diff(t_orig[:-1]) > 1e-9  # Check spacing between points

    # Use only unique points for interpolation
    t_orig_unique = t_orig[unique_mask]
    points_unique = points[unique_mask]

    # Check if enough unique points remain for interpolation
    if len(t_orig_unique) < 2:
        # Fallback: if not enough unique points, return the first unique point repeated
        return t_uniform, np.repeat(points_unique[:1], N, axis=0)

    # Interpolate x and y coordinates separately onto the uniform parameterization
    try:
        interp_func_x = interp1d(t_orig_unique, points_unique[:, 0], kind='linear', bounds_error=False,
                                 fill_value=(points_unique[0, 0], points_unique[-1, 0]))
        interp_func_y = interp1d(t_orig_unique, points_unique[:, 1], kind='linear', bounds_error=False,
                                 fill_value=(points_unique[0, 1], points_unique[-1, 1]))
        points_uniform = np.stack([interp_func_x(t_uniform), interp_func_y(t_uniform)], axis=-1)
    except ValueError:
        # Fallback in case of interpolation error
        points_uniform = np.repeat(points_unique[:1], N, axis=0)  # Repeat first point

    return t_uniform, points_uniform


def detect_anchors(curve):  # ... (implementation) ...
    """Detects anchor points (local extrema in y and endpoints) in a curve."""
    y = curve[:, 1];
    N_curve = len(curve)
    if N_curve < 2: return np.array([0]) if N_curve == 1 else np.array([])
    dy = np.diff(y);
    anchors = [0]  # Start point is always an anchor
    # Find points where the sign of the y-derivative changes (local extrema)
    for i in range(1, len(dy)):
        # Check for non-zero derivatives to avoid flat segments being marked as extrema
        if dy[i - 1] != 0 and dy[i] != 0 and dy[i - 1] * dy[i] < 0:
            anchors.append(i)  # Index corresponds to the point *after* the change
    anchors.append(N_curve - 1)  # End point is always an anchor
    # Ensure unique, sorted indices within valid range
    valid_anchors = np.unique(np.clip(anchors, 0, N_curve - 1)).astype(int)
    # Ensure at least start and end points if possible
    if len(valid_anchors) < 2 and N_curve > 1:
        valid_anchors = np.unique([0, N_curve - 1])
        valid_anchors = valid_anchors[valid_anchors < N_curve].astype(int)  # Clip again just in case
    return valid_anchors


def _calculate_curvature(curve_points):  # ... (implementation) ...
    """Calculates the curvature of a 2D curve defined by points."""
    if len(curve_points) < 3: return np.zeros(len(curve_points))
    # Use np.gradient for first and second derivatives
    dx_dt = np.gradient(curve_points[:, 0]);
    dy_dt = np.gradient(curve_points[:, 1])
    d2x_dt2 = np.gradient(dx_dt);
    d2y_dt2 = np.gradient(dy_dt)
    # Formula for curvature: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    # Add small epsilon to denominator to avoid division by zero
    curvature = numerator / (denominator + 1e-12)
    curvature[np.isnan(curvature)] = 0  # Handle potential NaNs
    return curvature


def _calculate_knots(num_interior_knots, N_uniform, uniform_curve=None,
                     knot_strategy='uniform'):  # ... (implementation) ...
    """Calculates interior knot locations based on different strategies."""
    if num_interior_knots < 0: return None  # Invalid input
    if num_interior_knots == 0: return np.array([])  # No interior knots needed

    knots = None
    t_param = np.linspace(0, 1, N_uniform, endpoint=True)  # Parameter space [0, 1]

    # --- Anchor Knot Strategy ---
    if knot_strategy == 'anchor' and uniform_curve is not None and len(uniform_curve) > 2:
        try:
            anchor_idx_for_knots = detect_anchors(uniform_curve)
            if len(anchor_idx_for_knots) >= 2:
                # Ensure indices are valid and get corresponding t values
                anchor_idx_valid = anchor_idx_for_knots[anchor_idx_for_knots < N_uniform]
                t_anchors = np.sort(np.unique(t_param[anchor_idx_valid]))

                # Distribute knots proportionally between anchor points
                if len(t_anchors) >= 2 and t_anchors[-1] - t_anchors[0] >= 1e-6:  # Need a valid range
                    segment_t_lengths = np.diff(t_anchors)
                    valid_segments = segment_t_lengths > 1e-9
                    if np.any(valid_segments):
                        valid_t_anchors_end = t_anchors[1:][valid_segments]  # End t-value of each valid segment
                        valid_segment_lengths = segment_t_lengths[valid_segments]
                        total_valid_t_length = np.sum(valid_segment_lengths)
                        if total_valid_t_length > 1e-9:
                            # Calculate cumulative distribution of segment lengths
                            cumulative_t_dist = np.cumsum(valid_segment_lengths) / total_valid_t_length
                            # Target proportions for knots within the anchor range
                            target_props = np.linspace(0, 1, num_interior_knots + 2)[1:-1]
                            # Interpolate to find knot locations in t-space
                            adaptive_knots = np.interp(target_props, cumulative_t_dist, valid_t_anchors_end)
                            # Validation: ensure knots are within (0, 1) and distinct
                            if len(adaptive_knots) == num_interior_knots and np.all(adaptive_knots > 1e-7) and np.all(
                                    adaptive_knots < 1 - 1e-7) and np.all(np.diff(np.unique(adaptive_knots)) > 1e-7):
                                knots = np.unique(adaptive_knots)
        except Exception:
            pass  # Fallback to uniform if anchor strategy fails

    # --- Curvature Knot Strategy ---
    elif knot_strategy == 'curvature' and uniform_curve is not None and len(uniform_curve) > 2:
        try:
            curvature = _calculate_curvature(uniform_curve)
            # Use absolute curvature as weight, add epsilon for stability
            weights = np.abs(curvature) + 1e-6
            total_weight = np.sum(weights)
            if total_weight > 1e-9:
                # Calculate cumulative weight distribution
                cum_weights = np.clip(np.cumsum(weights) / total_weight, 0, 1)
                # Target proportions for knots
                target_props = np.linspace(0, 1, num_interior_knots + 2)[1:-1]
                # Interpolate using cumulative weights to find knot t-values
                adaptive_knots = np.interp(target_props, cum_weights, t_param)
                adaptive_knots = np.clip(adaptive_knots, 1e-7, 1 - 1e-7)  # Clip to avoid boundaries
                adaptive_knots = np.unique(adaptive_knots)
                # Validation: ensure correct number and distinctness
                if len(adaptive_knots) >= num_interior_knots:  # Sometimes interp might yield fewer distinct points
                    # If too many, take a subset. If exact, use them.
                    if len(adaptive_knots) > num_interior_knots:
                        # Select evenly spaced indices from the potential knots
                        indices = np.round(np.linspace(0, len(adaptive_knots) - 1, num_interior_knots)).astype(int)
                        selected_knots = adaptive_knots[indices]
                        if np.all(np.diff(selected_knots) > 1e-7):  # Check distinctness again
                            knots = selected_knots
                    elif np.all(np.diff(adaptive_knots) > 1e-7):  # Exact number, check distinctness
                        knots = adaptive_knots
        except Exception as e:
            pass  # Fallback to uniform

    # --- Default or Fallback: Uniform knots ---
    if knots is None:
        # Place knots evenly in the (0, 1) interval
        knots = np.linspace(0, 1, num_interior_knots + 2)[1:-1]

    # Final check for empty result
    return knots if len(knots) > 0 else None


def point2para(points, method='dct', **kwargs):
    """Convert 2D discrete points to a curve parameter representation."""
    # --- Keep previous parameterization methods ---
    # --- Add 'rdp_k' ---
    points = np.asarray(points)
    if len(points) < 2: return None
    para_method = method
    if method == 'direct_k_cubic': method = 'direct_k'  # Treat cubic as variant of direct_k

    try:
        # --- direct_k: Uniform sampling after arc-length param ---
        if method == 'direct_k':
            K = kwargs.get('K')
            if K is None: raise ValueError("K required for 'direct_k'")
            t_uniform_param, uniform_curve = compute_arc_length_t(points);
            N_uniform = len(uniform_curve)
            if N_uniform == 0: return None
            actual_K = min(K, N_uniform);
            actual_K = max(1, actual_K)  # Ensure K is valid
            # Select K points uniformly from the arc-length parameterized curve
            t_select_param = np.linspace(0, 1, actual_K, endpoint=True)
            # Interpolate to get the selected points
            interp_func_x = interp1d(t_uniform_param, uniform_curve[:, 0], kind='linear', bounds_error=False,
                                     fill_value="extrapolate")
            interp_func_y = interp1d(t_uniform_param, uniform_curve[:, 1], kind='linear', bounds_error=False,
                                     fill_value="extrapolate")
            points_k = np.stack([interp_func_x(t_select_param), interp_func_y(t_select_param)], axis=-1)
            para = {'method': 'direct_k', 'points_k': points_k, 'K': actual_K}
            return para

        # --- dct: Discrete Cosine Transform coefficients ---
        elif method == 'dct':
            DCT_M = kwargs.get('DCT_M')
            if DCT_M is None: raise ValueError("DCT_M required for 'dct'")
            t_uniform, uniform_curve = compute_arc_length_t(points);
            N = len(uniform_curve)
            if N == 0: return None
            actual_DCT_M = min(DCT_M, N);
            actual_DCT_M = max(1, actual_DCT_M)  # Ensure M is valid
            # Apply DCT Type II
            x_full = dct(uniform_curve[:, 0], type=2, norm='ortho');
            y_full = dct(uniform_curve[:, 1], type=2, norm='ortho')
            # Truncate coefficients
            x_coeff_trunc = x_full[:actual_DCT_M];
            y_coeff_trunc = y_full[:actual_DCT_M]
            return {'method': 'dct', 'DCT_M': actual_DCT_M, 'N_orig_uniform': N, 'x_coeff': x_coeff_trunc,
                    'y_coeff': y_coeff_trunc}

        # --- anchor: LSQ Spline fitting with weighted anchors ---
        elif method == 'anchor':
            k = kwargs.get('k', 3);
            P = kwargs.get('P');
            w_anchor = kwargs.get('w_anchor', 1000);
            knot_strategy = kwargs.get('knot_strategy', 'uniform')
            if P is None or P <= 0: raise ValueError("'P' required for 'anchor' LSQ")
            t_uniform, uniform_curve = compute_arc_length_t(points);
            N_uniform = len(uniform_curve)
            if N_uniform == 0: return None
            # Detect anchor points
            anchor_idx = detect_anchors(uniform_curve)
            if len(anchor_idx) < 2:  # Ensure at least start and end
                anchor_idx = np.array([0, N_uniform - 1])
                anchor_idx = np.unique(anchor_idx[anchor_idx < N_uniform])  # Clip if needed
            if len(anchor_idx) == 0: return None  # Should not happen with above check
            anchors = uniform_curve[anchor_idx]  # Get anchor coordinates

            # Validate spline parameters
            if N_uniform <= k: return None  # Need more points than spline degree
            actual_k = k;
            actual_P = P
            if actual_P <= actual_k: actual_P = actual_k + 1  # Need P > k
            if actual_P > N_uniform: actual_P = N_uniform  # Cannot have more control points than data points
            num_interior_knots = actual_P - actual_k - 1

            # Calculate knots based on strategy
            knots = _calculate_knots(num_interior_knots, N_uniform, uniform_curve, knot_strategy)
            # Adjust k if no interior knots are possible/needed
            if knots is None and num_interior_knots < 0:
                actual_k = min(actual_k, actual_P - 1);
                actual_k = max(1, actual_k)  # Ensure k >= 1

            # Prepare data for LSQUnivariateSpline
            t_param_for_fitting = np.linspace(0, 1, N_uniform, endpoint=True)
            x = uniform_curve[:, 0];
            y = uniform_curve[:, 1]
            # Assign weights, higher weight to anchors
            w = np.ones(N_uniform)
            valid_anchor_idx = anchor_idx[anchor_idx < N_uniform]  # Ensure indices are within bounds
            w[valid_anchor_idx] = w_anchor

            # Fit splines
            try:
                spline_x = LSQUnivariateSpline(t_param_for_fitting, x, knots, w=w, k=actual_k, check_finite=False)
                spline_y = LSQUnivariateSpline(t_param_for_fitting, y, knots, w=w, k=actual_k, check_finite=False)
            except (ValueError, LinAlgError, TypeError) as e:
                return None  # Handle fitting errors

            para = {'method': 'anchor', 'spline_x': spline_x, 'spline_y': spline_y, 'k': actual_k, 'P': actual_P,
                    'anchors': anchors, 'w_anchor': w_anchor, 'knot_strategy': knot_strategy}
            return para

        # --- spline_lsq: LSQ Spline fitting without weighted anchors ---
        elif method == 'spline_lsq':
            k = kwargs.get('k', 3);
            P = kwargs.get('P');
            knot_strategy = kwargs.get('knot_strategy', 'uniform')
            if P is None: raise ValueError("'P' required for 'spline_lsq'")
            t_uniform, uniform_curve = compute_arc_length_t(points);
            N_uniform = len(uniform_curve)
            if N_uniform == 0: return None
            if N_uniform <= k: return None  # Need more points than spline degree

            # Validate spline parameters
            actual_k = k;
            actual_P = P
            if actual_P <= actual_k: actual_P = actual_k + 1  # Need P > k
            if actual_P > N_uniform: actual_P = N_uniform  # Cannot have more control points than data points
            num_interior_knots = actual_P - actual_k - 1

            # Calculate knots
            knots = _calculate_knots(num_interior_knots, N_uniform, uniform_curve, knot_strategy)
            # Adjust k if no interior knots are possible/needed
            if knots is None and num_interior_knots < 0:
                actual_k = min(actual_k, actual_P - 1);
                actual_k = max(1, actual_k)  # Ensure k >= 1

            # Prepare data (uniform weights)
            t_param_for_fitting = np.linspace(0, 1, N_uniform, endpoint=True)
            x = uniform_curve[:, 0];
            y = uniform_curve[:, 1]
            w = np.ones(N_uniform)  # Uniform weights

            # Fit splines
            try:
                spline_x = LSQUnivariateSpline(t_param_for_fitting, x, knots, w=w, k=actual_k, check_finite=False)
                spline_y = LSQUnivariateSpline(t_param_for_fitting, y, knots, w=w, k=actual_k, check_finite=False)
            except (ValueError, LinAlgError, TypeError) as e:
                return None  # Handle fitting errors

            para = {'method': 'spline_lsq', 'spline_x': spline_x, 'spline_y': spline_y, 'k': actual_k, 'P': actual_P,
                    'knot_strategy': knot_strategy}
            return para

        # --- dct_deviation: DCT of deviations from start-end baseline ---
        elif method == 'dct_deviation':
            K_target = kwargs.get(
                'K')  # K is the number of *points* desired in reconstruction (indirectly related to coeffs)
            if K_target is None: raise ValueError("K required for 'dct_deviation'")
            if K_target <= 2: return None  # Need at least start and end
            # Number of coefficients: Related to complexity, often M ~ 2K - 4
            M_coeffs = 2 * K_target - 4
            if M_coeffs <= 0: return None

            t_uniform, uniform_curve = compute_arc_length_t(points);
            N_uniform = len(uniform_curve)
            if N_uniform < 2: return None
            P_start = uniform_curve[0];
            P_end = uniform_curve[-1]
            V_baseline = P_end - P_start;
            baseline_len_sq = np.sum(V_baseline ** 2)

            # Handle zero-length baseline (start == end)
            if baseline_len_sq < 1e-12:
                # If start/end are same, deviation method doesn't make sense.
                # Maybe return a simplified representation like direct_k? Or None?
                return None  # Or {'method': 'dct_deviation', ... handle reconstruction appropriately}

            # Calculate perpendicular vector
            V_perp = np.array([-V_baseline[1], V_baseline[0]])
            V_perp_norm = V_perp / np.sqrt(baseline_len_sq)

            # Project points onto baseline and calculate perpendicular deviations
            t_param = np.linspace(0, 1, N_uniform, endpoint=True)
            P_baseline = P_start[None, :] + t_param[:, None] * V_baseline[None, :]
            Deviations = uniform_curve - P_baseline
            d_perp = np.dot(Deviations, V_perp_norm)  # Perpendicular distances

            # Apply DCT to deviations
            actual_M_coeffs = min(M_coeffs, len(d_perp))  # Ensure we don't ask for more coeffs than data
            if actual_M_coeffs <= 0: return None
            coeffs_full = dct(d_perp, type=2, norm='ortho')
            coeffs_trunc = coeffs_full[:actual_M_coeffs]

            para = {'method': 'dct_deviation', 'P_start': P_start, 'P_end': P_end, 'coeffs': coeffs_trunc,
                    'N_uniform': N_uniform, 'M_coeffs': actual_M_coeffs}
            return para

        # --- fft_complex: FFT of complex representation (x + iy) ---
        elif method == 'fft_complex':
            K_coeffs = kwargs.get('K')  # K is number of complex coefficients
            if K_coeffs is None: raise ValueError("K required for 'fft_complex'")
            t_uniform, uniform_curve = compute_arc_length_t(points);
            N_uniform = len(uniform_curve)
            if N_uniform == 0: return None
            actual_K = min(K_coeffs, N_uniform);
            actual_K = max(1, actual_K)  # Validate K

            # Create complex signal
            z = uniform_curve[:, 0] + 1j * uniform_curve[:, 1]
            # Apply FFT
            Z_full = np.fft.fft(z)
            # Truncate coefficients (take the first K)
            coeffs_trunc = Z_full[:actual_K]

            para = {'method': 'fft_complex', 'coeffs': coeffs_trunc, 'N_uniform': N_uniform, 'K_coeffs': actual_K}
            return para

        # --- NEW RDP_K METHOD ---
        elif method == 'rdp_k':
            K_target = kwargs.get('K')
            if K_target is None: raise ValueError("K must be provided for 'rdp_k'")
            if K_target < 2: raise ValueError("K must be >= 2 for 'rdp_k'")

            max_iterations = kwargs.get('rdp_max_iter', 15)  # Max iterations for epsilon search
            epsilon_tolerance = kwargs.get('rdp_epsilon_tol', 1e-5)  # Tolerance for epsilon search

            t_uniform, uniform_curve = compute_arc_length_t(points)
            N_uniform = len(uniform_curve)

            # Handle edge case: very few points
            if N_uniform <= 1:
                # Repeat the single point (or zero if empty) K_target times
                points_k = np.repeat(uniform_curve if N_uniform > 0 else np.zeros((1, 2)), K_target, axis=0)
                return {'method': 'rdp_k', 'simplified_points': points_k, 'K_actual': K_target, 'K_target': K_target}

            # Binary search for the epsilon that yields approximately K_target points
            eps_low = 0.0
            min_coords = np.min(uniform_curve, axis=0)
            max_coords = np.max(uniform_curve, axis=0)
            # Initial high epsilon: diagonal of the bounding box (a reasonable upper bound)
            eps_high = np.linalg.norm(max_coords - min_coords)
            if eps_high < 1e-9: eps_high = 1.0  # Avoid zero epsilon if all points are same

            best_eps_found = eps_high
            best_points = None
            best_count = 0

            for _ in range(max_iterations):
                eps_mid = (eps_low + eps_high) / 2.0
                # Avoid extremely small epsilon causing issues
                if eps_mid < 1e-10: break

                # Apply RDP with the current epsilon guess
                simplified = rdp(uniform_curve, epsilon=eps_mid)
                n_pts = len(simplified)

                # Adjust search range based on number of points found
                if n_pts <= K_target:
                    # Found a potential candidate or too few points, try smaller epsilon
                    best_eps_found = eps_mid
                    best_points = simplified
                    best_count = n_pts
                    eps_high = eps_mid  # Lower the upper bound
                else:
                    # Too many points, need larger epsilon
                    eps_low = eps_mid  # Raise the lower bound

                # Check convergence
                if (eps_high - eps_low) < epsilon_tolerance * eps_high:  # Relative tolerance
                    break

            # Ensure we have some points (at least start and end)
            if best_points is None or len(best_points) < 2:
                # Fallback if search failed or yielded < 2 points
                best_points = np.array([uniform_curve[0], uniform_curve[-1]])
                best_count = 2

            # Ensure exactly K_target points by adding points if necessary
            final_points = best_points
            current_count = len(final_points)

            if current_count < K_target:
                # Add points by subdividing the longest segments
                num_points_to_add = K_target - current_count

                # Use arc-length parameterization for smarter point insertion
                t_simplified, _ = compute_arc_length_t(final_points)

                # Calculate where to insert points based on uniform t-spacing
                t_insert = np.linspace(0, 1, K_target, endpoint=True)

                # Interpolate to get the final K points
                interp_x_add = interp1d(t_simplified, final_points[:, 0], kind='linear', bounds_error=False,
                                        fill_value="extrapolate")
                interp_y_add = interp1d(t_simplified, final_points[:, 1], kind='linear', bounds_error=False,
                                        fill_value="extrapolate")
                final_points = np.column_stack([interp_x_add(t_insert), interp_y_add(t_insert)])

            elif current_count > K_target:
                # This case should ideally be handled by the binary search finding the right epsilon.
                # If it still happens (e.g., due to tolerance), we might need to prune points.
                # Simplest prune: take K_target points evenly spaced by index from best_points
                indices = np.round(np.linspace(0, current_count - 1, K_target)).astype(int)
                final_points = best_points[indices]

            return {'method': 'rdp_k',
                    'simplified_points': final_points,
                    'K_actual': K_target,  # Should always be K_target now
                    'K_target': K_target}
            # --- End RDP_K ---

        else:
            raise ValueError(f"Unknown parameterization method: '{method}'.")
    except Exception as e:
        return None


def para2point(para, N_new, **kwargs):
    """Reconstruct 2D points from parameter representation. Returns None if reconstruction fails."""
    if para is None: return None
    # Allow overriding the method stored in para
    method = kwargs.get('method_override', para.get('method'))
    if method is None: return None

    try:
        # --- direct_k: Linear interpolation between K points ---
        if method == 'direct_k':
            if 'points_k' not in para or 'K' not in para: raise ValueError("Requires 'points_k'/'K'")
            points_k = para['points_k'];
            K = para['K']
            if K == 0: return np.empty((N_new, 2)) * np.nan  # Handle empty input
            if K < 2: return np.repeat(points_k, N_new, axis=0)  # Repeat if only one point
            # Interpolate between the K points
            t_param_k = np.linspace(0, 1, K, endpoint=True)
            t_param_new = np.linspace(0, 1, N_new, endpoint=True)
            interp_func_x = interp1d(t_param_k, points_k[:, 0], kind='linear', bounds_error=False,
                                     fill_value="extrapolate")
            interp_func_y = interp1d(t_param_k, points_k[:, 1], kind='linear', bounds_error=False,
                                     fill_value="extrapolate")
            result = np.column_stack([interp_func_x(t_param_new), interp_func_y(t_param_new)])
            if np.isnan(result).any(): return None  # Check for NaN results
            return result

        # --- direct_k_cubic: Cubic (or linear if K<4) interpolation between K points ---
        elif method == 'direct_k_cubic':
            if 'points_k' not in para or 'K' not in para: raise ValueError("Requires 'points_k'/'K'")
            points_k = para['points_k'];
            K = para['K']
            if K == 0: return np.empty((N_new, 2)) * np.nan
            if K < 2: return np.repeat(points_k, N_new, axis=0)
            # Choose interpolation kind based on number of points
            kind = 'cubic' if K >= 4 else 'linear'
            t_param_k = np.linspace(0, 1, K, endpoint=True)
            t_param_new = np.linspace(0, 1, N_new, endpoint=True)
            try:
                interp_func_x = interp1d(t_param_k, points_k[:, 0], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                interp_func_y = interp1d(t_param_k, points_k[:, 1], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                result = np.column_stack([interp_func_x(t_param_new), interp_func_y(t_param_new)])
                if np.isnan(result).any(): return None
                return result
            except ValueError as e:
                return None  # Handle interpolation errors

        # --- dct: Inverse DCT ---
        elif method == 'dct':
            if not all(k in para for k in ['DCT_M', 'N_orig_uniform', 'x_coeff', 'y_coeff']):
                raise ValueError("Missing required keys for 'dct' reconstruction")
            DCT_M = para['DCT_M'];
            N_orig_uniform = para['N_orig_uniform'];
            x_coeff_trunc = para['x_coeff'];
            y_coeff_trunc = para['y_coeff']
            # Pad coefficients with zeros if necessary
            x_coeff = np.zeros(N_orig_uniform);
            y_coeff = np.zeros(N_orig_uniform)
            len_coeffs = min(DCT_M, N_orig_uniform, len(x_coeff_trunc))  # Use actual length of stored coeffs
            x_coeff[:len_coeffs] = x_coeff_trunc[:len_coeffs]
            len_coeffs = min(DCT_M, N_orig_uniform, len(y_coeff_trunc))  # Use actual length of stored coeffs
            y_coeff[:len_coeffs] = y_coeff_trunc[:len_coeffs]
            # Apply inverse DCT Type II, requesting N_new points
            x_rec = idct(x_coeff, type=2, n=N_new, norm='ortho')
            y_rec = idct(y_coeff, type=2, n=N_new, norm='ortho')
            return np.column_stack([x_rec, y_rec])

        # --- anchor / spline_lsq: Evaluate the stored splines ---
        elif method in ['anchor', 'spline_lsq']:
            if 'spline_x' not in para or 'spline_y' not in para:
                raise ValueError(f"Missing 'spline_x' or 'spline_y' for '{method}' reconstruction")
            spline_x = para['spline_x'];
            spline_y = para['spline_y']
            t_new = np.linspace(0, 1, N_new, endpoint=True)
            # Evaluate splines at new points
            return np.column_stack([spline_x(t_new), spline_y(t_new)])

        # --- dct_deviation: Reconstruct deviations and add back to baseline ---
        elif method == 'dct_deviation':
            if not all(k in para for k in ['P_start', 'P_end', 'coeffs', 'N_uniform', 'M_coeffs']):
                raise ValueError("Missing required keys for 'dct_deviation' reconstruction")
            P_start = para['P_start'];
            P_end = para['P_end'];
            coeffs = para['coeffs'];
            N_uniform = para['N_uniform'];
            M_coeffs = para['M_coeffs']
            V_baseline = P_end - P_start;
            baseline_len_sq = np.sum(V_baseline ** 2)

            # Handle zero-length baseline
            if baseline_len_sq < 1e-12:
                return np.repeat(P_start[None, :], N_new, axis=0) if N_new > 0 else np.empty((0, 2))

            # Calculate perpendicular vector
            V_perp = np.array([-V_baseline[1], V_baseline[0]])
            V_perp_norm = V_perp / np.sqrt(baseline_len_sq)

            # Pad coefficients and apply inverse DCT
            coeffs_full = np.zeros(N_uniform)
            len_to_pad = min(M_coeffs, N_uniform, len(coeffs))  # Use actual length of stored coeffs
            coeffs_full[:len_to_pad] = coeffs[:len_to_pad]
            d_perp_rec = idct(coeffs_full, type=2, n=N_new, norm='ortho')  # Request N_new points

            # Recreate baseline points and add reconstructed deviations
            t_new = np.linspace(0, 1, N_new, endpoint=True)
            P_baseline_new = P_start[None, :] + t_new[:, None] * V_baseline[None, :]
            P_rec = P_baseline_new + d_perp_rec[:, None] * V_perp_norm[None, :]
            return P_rec

        # --- fft_complex: Inverse FFT ---
        elif method == 'fft_complex':
            if not all(k in para for k in ['coeffs', 'N_uniform', 'K_coeffs']):
                raise ValueError("Missing required keys for 'fft_complex' reconstruction")
            coeffs = para['coeffs'];
            N_uniform = para['N_uniform'];
            K_coeffs = para['K_coeffs']
            # Pad coefficients with zeros
            Z_full = np.zeros(N_uniform, dtype=np.complex128)
            len_to_pad = min(K_coeffs, N_uniform, len(coeffs))  # Use actual length of stored coeffs
            Z_full[:len_to_pad] = coeffs[:len_to_pad]
            # Apply inverse FFT, requesting N_new points
            z_rec = np.fft.ifft(Z_full, n=N_new)
            x_rec = z_rec.real;
            y_rec = z_rec.imag
            return np.column_stack([x_rec, y_rec])

        # --- RDP_K METHOD (Reconstruction from simplified points) ---
        elif method == 'rdp_k':
            if 'simplified_points' not in para: raise ValueError("Requires 'simplified_points'")
            simplified_points = para['simplified_points']
            N_simplified = len(simplified_points)

            if N_simplified == 0: return np.empty((N_new, 2)) * np.nan
            if N_simplified < 2: return np.repeat(simplified_points, N_new, axis=0)

            # Interpolate between simplified points using arc-length param
            # Use the same compute_arc_length_t but only get the t parameter
            t_rdp, simplified_points_uniform = compute_arc_length_t(
                simplified_points)  # Get t and potentially uniform points

            # Ensure t_rdp corresponds to the original simplified_points if compute_arc_length_t modified them
            # Recompute t based on the *original* simplified points for accuracy
            diffs_rdp = np.diff(simplified_points, axis=0)
            segment_lengths_rdp = np.sqrt(np.sum(diffs_rdp ** 2, axis=1))
            s_rdp = np.concatenate(([0], np.cumsum(segment_lengths_rdp)))
            total_length_rdp = s_rdp[-1]
            if total_length_rdp < 1e-9:
                t_rdp_accurate = np.linspace(0, 1, N_simplified, endpoint=True)
            else:
                t_rdp_accurate = s_rdp / total_length_rdp
                t_rdp_accurate[0] = 0.0;
                t_rdp_accurate[-1] = 1.0  # Ensure bounds

            # Remove duplicates in t_rdp_accurate for interpolation
            unique_mask_rdp = np.concatenate(([True], np.diff(t_rdp_accurate) > 1e-9))
            t_rdp_unique = t_rdp_accurate[unique_mask_rdp]
            simplified_points_unique = simplified_points[unique_mask_rdp]

            if len(t_rdp_unique) < 2:  # Fallback if too few unique points
                return np.repeat(simplified_points_unique[:1], N_new, axis=0)

            t_new = np.linspace(0, 1, N_new, endpoint=True)
            try:
                # Choose interpolation kind - linear is often sufficient for RDP output
                kind = 'linear'  # 'cubic' if len(t_rdp_unique) >= 4 else 'linear'

                interp_func_x = interp1d(t_rdp_unique, simplified_points_unique[:, 0], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                interp_func_y = interp1d(t_rdp_unique, simplified_points_unique[:, 1], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                result = np.column_stack([interp_func_x(t_new), interp_func_y(t_new)])
                if np.isnan(result).any():
                    return None
                return result
            except ValueError as e:
                return None
        # --- End RDP_K ---

        # --- NEW RDP_K_WITHOD METHOD ---
        elif method == 'rdp_k_withod':
            # 1. Input Validation
            required_keys = ['simplified_points', 'start_point', 'end_point']
            if not all(key in para for key in required_keys):
                raise ValueError(f"Requires {required_keys} in para for 'rdp_k_withod'")

            simplified_points = para['simplified_points']
            # Ensure start/end points are numpy arrays for comparison
            start_point = np.array(para['start_point'])
            end_point = np.array(para['end_point'])
            N_simplified = len(simplified_points)

            if N_simplified == 0:
                return np.empty((N_new, 2)) * np.nan

            # 2. Find Indices of Start and End Points
            # Use np.isclose for robust floating-point comparison
            start_indices = np.where(np.isclose(simplified_points, start_point).all(axis=1))[0]
            end_indices = np.where(np.isclose(simplified_points, end_point).all(axis=1))[0]

            if len(start_indices) == 0:
                return None
            if len(end_indices) == 0:
                return None

            # Take the first occurrence if multiple matches (unlikely with RDP output)
            start_idx = start_indices[0]
            end_idx = end_indices[0]

            if start_idx == end_idx:
                # If start and end points are the same, repeat that point
                return np.repeat(simplified_points[start_idx:start_idx + 1], N_new, axis=0)
            elif start_idx > end_idx:
                # Allow reversing if user provides them in wrong order? Optional.
                # start_idx, end_idx = end_idx, start_idx # Uncomment to swap if needed
                return None  # Or swap as above

            # 3. Extract the Segment
            segment_points = simplified_points[start_idx: end_idx + 1]
            N_segment = len(segment_points)

            if N_segment < 2:  # Should not happen if start_idx < end_idx, but safe check
                return np.repeat(segment_points, N_new, axis=0) if N_segment == 1 else np.empty((N_new, 2)) * np.nan

            # 4. Reconstruct using the Segment via Arc-Length Parameterized Interpolation
            # Compute arc-length parameterization for the segment
            diffs_seg = np.diff(segment_points, axis=0)
            segment_lengths_seg = np.sqrt(np.sum(diffs_seg ** 2, axis=1))
            s_seg = np.concatenate(([0], np.cumsum(segment_lengths_seg)))
            total_length_seg = s_seg[-1]
            if total_length_seg < 1e-9:
                t_segment_accurate = np.linspace(0, 1, N_segment, endpoint=True)
            else:
                t_segment_accurate = s_seg / total_length_seg
                t_segment_accurate[0] = 0.0;
                t_segment_accurate[-1] = 1.0  # Ensure bounds

            # Remove duplicates for interpolation robustness
            unique_mask_seg = np.concatenate(([True], np.diff(t_segment_accurate) > 1e-9))
            t_segment_unique = t_segment_accurate[unique_mask_seg]
            segment_points_unique = segment_points[unique_mask_seg]

            if len(t_segment_unique) < 2:  # Fallback
                return np.repeat(segment_points_unique[:1], N_new, axis=0)

            # Target parameter values for the new points
            t_new = np.linspace(0, 1, N_new, endpoint=True)

            try:
                # Choose interpolation kind (linear often good for RDP segments)
                kind = 'linear'  # 'cubic' if len(t_segment_unique) >= 4 else 'linear'

                interp_func_x = interp1d(t_segment_unique, segment_points_unique[:, 0], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                interp_func_y = interp1d(t_segment_unique, segment_points_unique[:, 1], kind=kind, bounds_error=False,
                                         fill_value="extrapolate")
                result = np.column_stack([interp_func_x(t_new), interp_func_y(t_new)])

                # Check for NaNs
                if np.isnan(result).any():
                    # Optionally try linear fallback if cubic failed
                    if kind == 'cubic':
                        interp_func_x = interp1d(t_segment_unique, segment_points_unique[:, 0], kind='linear',
                                                 bounds_error=False, fill_value="extrapolate")
                        interp_func_y = interp1d(t_segment_unique, segment_points_unique[:, 1], kind='linear',
                                                 bounds_error=False, fill_value="extrapolate")
                        result = np.column_stack([interp_func_x(t_new), interp_func_y(t_new)])
                        if np.isnan(result).any():
                            return None  # Give up if linear also fails
                    else:
                        return None  # Linear already failed
                return result

            except ValueError as e:
                return None
        # --- End RDP_K_WITHOD ---

        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

    except Exception as e:
        return None

def para2point_batch(trajs_array, N_new, method='rdp_k'):
    """Convert a batch of parameterized trajectories back to point sequences.

    Args:
        trajs_array: Trajectory array with shape [batch_size, M, 2].
        N_new: Target trajectory length.
        method: Parameterization method name.

    Returns:
        Reconstructed trajectory array with shape [batch_size, N_new, 2].
    """
    batch_size = trajs_array.shape[0]
    batch_reconstructed = []

    for i in range(batch_size):
        para_dict = {
            'method': method,
            'simplified_points': trajs_array[i].reshape(-1, 2),
            'K_actual': None,
            'K_target': None
        }
        reconstructed = para2point(para_dict, N_new=N_new, method_override=method)
        if reconstructed is None:
            print(f"Warning: failed to reconstruct trajectory {i}; using zeros instead")
            reconstructed = np.zeros((N_new, 2))
        batch_reconstructed.append(reconstructed)

    return np.array(batch_reconstructed)

import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to Python path to import your modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import the transformer functions
