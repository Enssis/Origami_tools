__all__ = [
    "_svg_utils", # type: ignore
    "_types"   # type: ignore
]

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