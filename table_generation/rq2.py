import pandas as pd
df_clean = pd.read_csv('../data_csvs/rq1_results.csv')
agg_clean = df_clean.groupby(['Model', 'Lang', 'Smell']).mean().reset_index()
df_adv = pd.read_csv('../data_csvs/rq2_results.csv')
agg_adv = df_adv.groupby(['Model', 'Lang', 'Smell', 'Mode']).mean().reset_index()

def get_arrow(a_val, c_val):
    a_str = f'{a_val:.3f}'
    c_str = f'{c_val:.3f}'
    if a_str == c_str:
        return '-'
    elif a_val < c_val:
        return '$\\downarrow$'
    else:
        return '$\\uparrow$'

def build_rq2_latex(metric, is_f1):
    latex = f"\\begin{{table}}[!ht]\n\\caption{{Model {metric} Degradation on Adversarial Test Samples (By Language and Code Smell)}}\n\\label{{table_rq2_{('new' if is_f1 else 'auc')}}}\n\\centering\\resizebox{{\\textwidth}}{{!}}{{\n\\begin{{tabular}}{{lll|cc|cc|cc|cc}}\n\\hline\n\\cellcolor{{blue!5}} & \\cellcolor{{blue!5}} & \\cellcolor{{blue!5}} & \\multicolumn{{2}}{{c|}}{{\\cellcolor{{blue!5}}\\textbf{{CM}}}} & \\multicolumn{{2}}{{c|}}{{\\cellcolor{{blue!5}}\\textbf{{CC}}}} & \\multicolumn{{2}}{{c|}}{{\\cellcolor{{blue!5}}\\textbf{{FE}}}} & \\multicolumn{{2}}{{c}}{{\\cellcolor{{blue!5}}\\textbf{{MA}}}} \\\\\n\\cellcolor{{blue!5}} \\textbf{{Model}} & \\cellcolor{{blue!5}} \\textbf{{Lang}} & \\cellcolor{{blue!5}} \\textbf{{Attack}} & \\cellcolor{{blue!5}} \\textbf{{Clean}} & \\cellcolor{{blue!5}} \\textbf{{Attacked}} & \\cellcolor{{blue!5}} \\textbf{{Clean}} & \\cellcolor{{blue!5}} \\textbf{{Attacked}} & \\cellcolor{{blue!5}} \\textbf{{Clean}} & \\cellcolor{{blue!5}} \\textbf{{Attacked}} & \\cellcolor{{blue!5}} \\textbf{{Clean}} & \\cellcolor{{blue!5}} \\textbf{{Attacked}} \\\\\n\\hline\n"
    modes = ['Semantic', 'Structural', 'Hybrid', 'NLP']
    for mod in ['RNN', 'CodeBERT', 'AE', 'Qwen']:
        for lang in ['CSharp', 'Java']:
            for mode in modes:
                row_str = f'\\cellcolor{{blue!5}} {mod} & \\cellcolor{{blue!5}} {lang} & \\cellcolor{{blue!5}} {mode} & '
                cells = []
                for smell in ['CM', 'CC', 'FE', 'MA']:
                    c_row = agg_clean[(agg_clean['Model'] == mod) & (agg_clean['Lang'] == lang) & (agg_clean['Smell'] == smell)]
                    c_val = c_row['F1' if is_f1 else 'AUC'].values[0]
                    a_row = agg_adv[(agg_adv['Model'] == mod) & (agg_adv['Lang'] == lang) & (agg_adv['Smell'] == smell) & (agg_adv['Mode'] == mode)]
                    a_val = a_row['F1' if is_f1 else 'AUC'].values[0]
                    arrow = get_arrow(a_val, c_val)
                    cells.append(f'\\cellcolor{{blue!5}}{c_val:.3f} & \\cellcolor{{blue!5}}{a_val:.3f} {arrow}')
                row_str += ' & '.join(cells) + ' \\\\\n'
                latex += row_str
            latex += '\\hline\n'
    latex += '\\end{tabular}}\n\\end{table}\n'
    return latex
print(build_rq2_latex('F1', True).strip())
print()
print(build_rq2_latex('AUC', False).strip())