from ._svg_utils import hsv_to_hex, rgb_to_hex, hex_to_rgb, simplifed_hex, mm_str, save_svg
from ._types import Number, Group



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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
    return lambda x : (fun(x + deps) - fun(x - deps)) / (2 * deps)

def min_search(start : Number, fun, deps=0.001, tol=0.001, max_iter=100, graph_x_deriv=False, anim_graph=False, graph_limits = [-10, 10]):
    x = start
    def deriv (x):
        return (fun(x + deps) - fun(x - deps)) / (2 * deps)
    
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


    def animated_graph(xs):
        plt.ion()
        fig, ax = plt.subplots()
        x = np.linspace(graph_limits[0], graph_limits[1], 100)
        y = [fun(x_i) for x_i in x]
        ax.plot(x, y, "b")
        point, = ax.plot([], [], "ro")
        for i in range(len(xs)):
            point.set_data(xs[:i], [fun(x_i) for x_i in xs[:i]])
            plt.pause(0.1)
        plt.ioff()
        plt.show()
    
    nan_detected = False
    xs = []
    ders = []
    etas = []
    d = deriv(x) / max(fun(x), 1)
    dn_1 = 0
    xn_1 = 0
    eta = 0.01
    for i in range(max_iter):
        if graph_x_deriv or anim_graph:
            xs.append(x)
            ders.append(abs(d))
            etas.append(eta)

        # print("iteration", i, "x =", x, "deriv =", d)
        if abs(d) < tol:
            if graph_x_deriv:
                dual_graph(xs, ders)
            if anim_graph:
                animated_graph(xs)
            return x
        
        if i > 1:
            diff_d = d - dn_1
            eta = abs((x - xn_1) * diff_d) / (np.linalg.norm(diff_d)**2)
            # print("diff_g =", diff_g, "x - xn_1 =", x - xn_1, "eta =", eta)
            # print("eta =", eta)
        else :
            eta = 0.01
        if np.isnan(x).any() or np.isnan(d).any():
            print("NaN detected, stopping optimization. starting point:", start)
            x = xn_1
            d = dn_1
            nan_detected = True
            break
        # eta = 0.001
        xn_1 = x
        dn_1 = d
        x -= eta * d
        d = deriv(x)
    print("Max iterations reached without convergence, derivative:", d)
    if graph_x_deriv:
        dual_graph(xs, ders)
    if anim_graph:
        animated_graph(xs)
    if nan_detected:
        return np.nan
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

    points_per_segment = [int(num_points * ratio) for ratio in segment_ratios]
    # Adjust the number of points to ensure we have exactly num_points
    while sum(points_per_segment) < num_points:
        for i in range(len(points_per_segment)):
            if sum(points_per_segment) < num_points:
                points_per_segment[i] += 1

    result = []
    segments = [[] for _ in range(len(points)-1)]
    for i in range(len(points)-1):
        start, end = np.array(points[i]), np.array(points[i+1])
        for j in range(points_per_segment[i]):
            t = j / points_per_segment[i]
            if segmented:
                segments[i].append(start + t * (end - start))
            result.append(start + t * (end - start))
    if segmented:
        return result, segments
    return result


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
    "deriv_fun"
]