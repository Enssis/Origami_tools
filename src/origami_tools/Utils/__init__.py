from ._svg_utils import hsv_to_hex, rgb_to_hex, hex_to_rgb, simplifed_hex, mm_str, save_svg, rgb_vals_to_hex, hex_to_hsv
from ._types import Number, Group



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.optimize import line_search

def csv_to_latex(csv_file : str, latex_file : str):
    """
        Convert a csv file to a latex file
    """
    if not csv_file.endswith(".csv"):
        raise ValueError("The file must be a csv file")
    if not latex_file.endswith(".tex"):
        raise ValueError("The file must be a latex file")

    with open(csv_file, "r") as f:
        lines = f.readlines()
    nb_col = len(lines[0].split(","))
    with open(latex_file, "w") as f:
        f.write('''\
\\begin{{table}}[H]
\\centering
\\resizebox{{\\textwidth}}{{!}}{{%
    \\begin{{tabular}}{{@{{}}{c_cols}@{{}}}}
    \\toprule   
'''.format(c_cols="c" * nb_col))
        for i in range(len(lines)):
            line = lines[i]
            if i == 0:
                temp_line = line[:-1].split(",")
                line = ""
                for j in range(nb_col):
                    line += "\\textbf{" + temp_line[j] + "}"
                    if j != nb_col - 1:
                        line += " & "
                line += "\n \\midrule"
            f.write("\t" + line.replace(",", " & ").replace("\n", " \\\\\n").replace("_", "\\_").replace("#", "\\#"))
        
        name = latex_file.split("/")[-1].split(".")[0]
        
        f.write('''\
    \\bottomrule
    \\end{{tabular}}%
}
\\caption{caption}
\\label{label}
\\end{{table}}
'''.format(caption=name, label="fig:" + name))
        

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def deriv_fun(fun, deps=0.01):
    return lambda x : (fun(x + deps) - fun(x)) / (deps)

def gradient(fun, deps=0.01):
    def grad(x):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            dep = np.zeros_like(x)
            dep[i] = deps
            grad[i] = (fun(x + dep) - fun(x - dep)) / (2 * deps)
        return grad
    return grad

# def adam(objective, derivative, start, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
# 	solutions = list()
# 	# generate an initial point
# 	x = start
# 	score = objective(x[0], x[1])
# 	# initialize first and second moments
# 	m = [0.0 for _ in range(bounds.shape[0])]
# 	v = [0.0 for _ in range(bounds.shape[0])]
# 	# run the gradient descent updates
# 	for t in range(n_iter):
# 		# calculate gradient g(t)
# 		g = derivative(x[0], x[1])
# 		# build a solution one variable at a time
# 		for i in range(bounds.shape[0]):
# 			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
# 			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
# 			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
# 			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
# 			# mhat(t) = m(t) / (1 - beta1(t))
# 			mhat = m[i] / (1.0 - beta1**(t+1))
# 			# vhat(t) = v(t) / (1 - beta2(t))
# 			vhat = v[i] / (1.0 - beta2**(t+1))
# 			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
# 			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
# 		# evaluate candidate point
# 		score = objective(x[0], x[1])
# 		# keep track of solutions
# 		solutions.append(x.copy())
# 		# report progress
# 		print('>%d f(%s) = %.5f' % (t, x, score))
# 	return solutions


def minimize_search(function, grad = None, start : None | list[Number] = [], bounds : list[tuple[Number, Number]] = [(-10, 10), (-10, 10)], max_iter : int = 100, alpha = 0.01, beta1 = 0.9, beta2 = 0.999, eps=1e-8, tol=1e-9):

    if grad is None:
        grad = gradient(function)

    if start is None or len(start) == 0:
        start = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    
def adam_optimize(f, grad_f, theta0, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iter=1000):
    theta = theta0
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, max_iter+1):
        g = grad_f(theta)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
    return theta


def min_search(x0 : Number, f, g = None,  eps=0.001, tol=0.001, max_iter=100, eta=0.001, graph_x_deriv=False, anim_graph=False, graph_limits = [-10, 10], verbose=False):
    x = x0
    
    if g is not None:
        def deriv(x):
            return g(x)
    else:
        def deriv (x):
            der = (f(x + eps) - f(x - eps)) / (2 * eps)
            if np.isnan(der):
                print("NaN detected in derivative, stopping optimization. current point:", x, "fun(x + deps):", f(x + eps), "fun(x - deps):", f(x - eps))
            return der
    
    def dual_graph(xs, ders):
        # On récupère le système d'axes (ax) et on l'utilise pour tracer la fonction carré.
        fig, ax = plt.subplots()
        ax.plot(range(len(xs)), xs, "b")
        ax.set_xlabel("iteration", fontsize=14)
        ax.set_ylabel("x", color="blue", fontsize=14)

        # On produit un second système d'axes à partir du premier et on l'utilise pour tracer la fonction sinus.
        ax2 = ax.twinx()
        ax2.plot(range(len(ders)), ders, "r")
        ax2.set_ylabel("derivativ", color="red", fontsize=14)

        # On prépare la légende pour nos deux courbes.
        lines = [ax.get_lines()[0], ax2.get_lines()[0]]
        plt.legend(lines, ["x", "deriv"], loc="upper center")

        # Et on affiche le tout.
        plt.show()


    def tests_graph(xs):
        # plt.ion()
        # fig, ax = plt.subplots()
        # x = np.linspace(graph_limits[0], graph_limits[1], 100)
        # y = [fun(x_i) for x_i in x]
        # ax.plot(x, y, "b")
        # point, = ax.plot([], [], "ro")
        # for i in range(len(xs)):
        #     point.set_data(xs[:i], [fun(x_i) for x_i in xs[:i]])
        #     plt.pause(0.1)
        # plt.ioff()
        # plt.show()
        x = np.linspace(graph_limits[0], graph_limits[1], 100)
        y = [f(x_i) for x_i in x]
        plt.plot(x, y, "b")
        plt.plot(xs[:-1], [f(x_i) for x_i in xs[:-1]], "ro")
        plt.plot(xs[-1], f(xs[-1]), "bo")
        plt.show()
    
    nan_detected = False
    xs = []
    ders = []
    etas = []
    d = deriv(x) #/ max(fun(x), 1)
    dn_1 = 0
    xn_1 = 0
    cache = 0
    eps = 1e-8
    E_g_sq = 0
    E_Dx_sq = 0
    etai= eta

    for i in range(max_iter):
        
        xs.append(x)
        ders.append(abs(d))
        etas.append(etai)

        # print("iteration", i, "x =", x, "deriv =", d)
        if abs(d) < tol:
            if graph_x_deriv:
                dual_graph(xs, ders)
            if anim_graph:
                tests_graph(xs)
            return x
        
        if i > 1:
            diff_d = d - dn_1
            Dx = (x - xn_1)
            if np.isnan(diff_d) or np.linalg.norm(diff_d) == 0:
                if verbose:
                    print("NaN or zero detected in diff_d, using default eta. current point:", x, "derivative:", d, "previous derivative:", dn_1)
                    print("eta =", etai)
                etai = eta
            else:
                etai = abs(Dx) / abs(diff_d)
            # etai = np.linalg.norm(Dx)**2 / abs(Dx * diff_d)

        else :
            etai = eta
        if np.isnan(x).any() or np.isnan(d).any():
            if verbose:
                print("NaN detected, stopping optimization. starting point:", x0)
                print("last point before NaN:", xn_1, "derivative at last point:", dn_1, "current point:", x, "current derivative:", d, "diff_d =", d - dn_1)
            x = xn_1
            d = dn_1
            # nan_detected = True
            break
        # eta = 0.001
        xn_1 = x
        dn_1 = d
        x -= etai * d
        d = deriv(x)
        # E_g_sq = beta * E_g_sq + (1-beta) * d**2

        # RMS_g = np.sqrt(E_g_sq + eps)
        # RMS_Dx = np.sqrt(E_Dx_sq + eps)
        # Dx = - (RMS_Dx / RMS_g) * d

        # E_Dx_sq = beta * E_Dx_sq + (1-beta) * Dx**2

        # # cache += beta * cache + (1-beta) * d**2
        # x += eta * d #Dx #eta * d / np.sqrt(cache + eps)
    if verbose:
        print("Max iterations reached without convergence, derivative:", d)
    if graph_x_deriv:
        dual_graph(xs, ders)
    if anim_graph:
        tests_graph(xs)
    # if nan_detected:
    #     return np.nan
    return x

def min_search_grad(start, fun, deps=0.01, tol=0.001, max_iter=100, graph=False):
    x = start

    def gradient(x):
        # print("gradx :", x)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            dep = np.zeros_like(x)
            dep[i] = deps
            grad[i] = (fun(x + dep) - fun(x - dep)) / (2 * deps)
        return grad
    
    def dual_graph(xs, ders):
        # On récupère le système d'axes (ax) et on l'utilise pour tracer la fonction carré.
        for i in range(len(xs)):
            fig, ax = plt.subplots()
            ax.plot(range(len(xs[i])), xs[i], "b")
            ax.set_xlabel("iteration", fontsize=14)
            ax.set_ylabel("x_" + str(i), color="blue", fontsize=14)

            # On produit un second système d'axes à partir du premier et on l'utilise pour tracer la fonction sinus.
            ax2 = ax.twinx()
            ax2.plot(range(len(ders[i])), ders[i], "r")
            ax2.set_ylabel("derivativ", color="red", fontsize=14)

            # On prépare la légende pour nos deux courbes.
            lines = [ax.get_lines()[0], ax2.get_lines()[0]]
            plt.legend(lines, ["x", "deriv"], loc="upper center")

            # Et on affiche le tout.
            plt.show()
    
    xs = [[] for _ in range(len(x))]
    ders = [[] for _ in range(len(x))]
    g = gradient(x)

    xn_1 = np.zeros_like(x)
    gn_1 = np.zeros_like(x)
    for i in range(max_iter):
        if graph:
            for j in range(len(x)):
                xs[j].append(x[j])
                ders[j].append(g[j])
        if np.linalg.norm(g) < tol:
            if graph:
                dual_graph(xs, ders)
            return x
        
        if i > 1:
            diff_g = g - gn_1
            eta = abs((x - xn_1).T @ diff_g) / (np.linalg.norm(diff_g)**2)
            # print("diff_g =", diff_g, "x - xn_1 =", x - xn_1, "eta =", eta)
            # print("eta =", eta)
        else :
            eta = 0.1

        if np.isnan(x).any() or np.isnan(g).any():
            print("NaN detected, stopping optimization.")
            x = xn_1
            g = gn_1
            break
        xn_1 = np.copy(x)
        gn_1 = np.copy(g)
        
        # eta = 0.001
        x = [x - eta * g_i for x, g_i in zip(x, g)]
        # print(x, g)
        g = gradient(x)
    print("Max iterations reached without convergence, derivative:", g)
    if graph:
        dual_graph(xs, ders)
    return x


def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1.0, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                        mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)
    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n]) # type: ignore
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2])) # type: ignore
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, # type: ignore
            **arrow_kw)# type: ignore
        axes.add_patch(p)
        arrows.append(p)
    return arrows


def multi_point_linspace(points : list[Number], num_points : int, segmented = False):
    """
        Generate a list of points that are evenly spaced between the given points.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required")
    
    segment_lengths = np.array([abs(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1)])
    total_length = sum(segment_lengths)
    segment_ratios = segment_lengths / total_length

    points_per_segment = [int((num_points-1) * ratio) for ratio in segment_ratios]
    # Adjust the number of points to ensure we have exactly num_points
    while sum(points_per_segment) < num_points-1:
        for i in range(len(points_per_segment)):
            if sum(points_per_segment) < num_points-1:
                points_per_segment[i] += 1

    result = []
    segments = [[] for _ in range(len(points)-1)]
    for i in range(len(points)-1):
        start, end = np.array(points[i]), np.array(points[i+1])
        for j in range(points_per_segment[i]):
            t = j / points_per_segment[i]
            if segmented:
                segments[i].append(start + t * (end - start)) # type: ignore
            result.append(start + t * (end - start))
    result.append(points[-1])
    if segmented:
        return result, segments
    return result


def gravity_spaced(p_start : Number, p_end : Number, p_grav : list[Number], num_points : int, g=10.0):
    """
        Generate a list of points that are spaced between the given points, with a higher density around the gravity point.
    """
    
    start_spaced = np.linspace(p_start, p_end, num_points)
    scale = abs(p_start - p_end)

    def gravity_func(p):
        return sum([(p_g - p) * (1-2 /np.pi * np.atan(g / scale * abs(p_g - p)))**2 for p_g in p_grav])
    
    for i in range(1, len(start_spaced)-1):
        # print("p:", start_spaced[i], "gravity:", gravity_func(start_spaced[i]), "dist :",  [abs(p_g - start_spaced[i]) for p_g in p_grav], "after gravity:", start_spaced[i] + gravity_func(start_spaced[i]))
        start_spaced[i] += gravity_func(start_spaced[i])

    return start_spaced.tolist()




__all__ = [
    "csv_to_latex",
    "deg2rad",
    "rad2deg",
    "hsv_to_hex",
    "rgb_to_hex",
    "hex_to_rgb",
    "simplifed_hex",
    "mm_str",
    "save_svg",
    "Number",
    "Group", 
    "min_search",
    "min_search_grad", 
    "multi_point_linspace",
    "deriv_fun",
    "hex_to_hsv",
    "gravity_spaced"
]