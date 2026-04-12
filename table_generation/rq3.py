import pandas as pd
df_clean = pd.read_csv('../data_csvs/rq1_results.csv')
agg_clean = df_clean.groupby(['Model', 'Lang', 'Smell']).mean().reset_index()
df_adv = pd.read_csv('../data_csvs/rq3_results.csv')
agg_adv = df_adv.groupby(['Model', 'Lang', 'Smell', 'Ratio']).mean().reset_index()

def get_arrow(a_val, c_val):
    a_str = f'{a_val:.3f}'
    c_str = f'{c_val:.3f}'
    if a_str == c_str:
        return '-'
    elif a_val < c_val:
        return '$\\downarrow$'
    else:
        return '$\\uparrow$'

def build_rq3_latex(metric, is_f1):
    caption = 'Clean-Test F1 After Adversarial Training at Different Adversarial Ratios' if is_f1 else 'Clean-Test AUC-ROC After Adversarial Training at Different Adversarial Ratios'
    label = '\\label{table_rq3_new}' if is_f1 else '\\label{table_rq3_auc}'
    latex = f'\\begin{{table}}[H]\n\\caption{{{caption}}}\n{label}\n\\centering\\footnotesize\n\\resizebox{{\\textwidth}}{{!}}{{\n\\begin{{tabular}}{{llcccc|cccc}}\n\\hline\n\\cellcolor{{blue!5}} & \\cellcolor{{blue!5}} & \\multicolumn{{4}}{{c|}}{{\\cellcolor{{blue!5}}\\textbf{{CM}}}} & \\multicolumn{{4}}{{c}}{{\\cellcolor{{blue!5}}\\textbf{{CC}}}} \\\\\n\\cellcolor{{blue!5}}\\textbf{{Model}} & \\cellcolor{{blue!5}}\\textbf{{Lang}} & \\cellcolor{{blue!5}}\\textbf{{Base}} & \\cellcolor{{blue!5}}\\textbf{{10\\%}} & \\cellcolor{{blue!5}}\\textbf{{30\\%}} & \\cellcolor{{blue!5}}\\textbf{{50\\%}} & \\cellcolor{{blue!5}}\\textbf{{Base}} & \\cellcolor{{blue!5}}\\textbf{{10\\%}} & \\cellcolor{{blue!5}}\\textbf{{30\\%}} & \\cellcolor{{blue!5}}\\textbf{{50\\%}} \\\\\n\\hline\n'
    for mod in ['RNN', 'CodeBERT', 'AE', 'Qwen']:
        for lang in ['CSharp', 'Java']:
            row = f'\\cellcolor{{blue!5}}{mod} & \\cellcolor{{blue!5}}{lang} '
            for smell in ['CM', 'CC']:
                c_row = agg_clean[(agg_clean['Model'] == mod) & (agg_clean['Lang'] == lang) & (agg_clean['Smell'] == smell)]
                base = c_row['F1' if is_f1 else 'AUC'].values[0]
                row += f'& \\cellcolor{{blue!5}}{base:.3f} '
                for ratio in ['10%', '30%', '50%']:
                    a_row = agg_adv[(agg_adv['Model'] == mod) & (agg_adv['Lang'] == lang) & (agg_adv['Smell'] == smell) & (agg_adv['Ratio'] == ratio)]
                    adv_val = a_row['F1' if is_f1 else 'AUC'].values[0]
                    arrow = get_arrow(adv_val, base)
                    if arrow == '-':
                        row += f'& \\cellcolor{{blue!5}}{adv_val:.3f} - '
                    else:
                        row += f'& \\cellcolor{{blue!5}}{adv_val:.3f} {arrow} '
            row += '\\\\\n'
            latex += row
        latex += '\\hline\n'
    latex += '\\end{tabular}}\n'
    latex += '\\vspace{2mm}\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{llcccc|cccc}\n\\hline\n\\cellcolor{blue!5} & \\cellcolor{blue!5} & \\multicolumn{4}{c|}{\\cellcolor{blue!5}\\textbf{FE}} & \\multicolumn{4}{c}{\\cellcolor{blue!5}\\textbf{MA}} \\\\\n\\cellcolor{blue!5}\\textbf{Model} & \\cellcolor{blue!5}\\textbf{Lang} & \\cellcolor{blue!5}\\textbf{Base} & \\cellcolor{blue!5}\\textbf{10\\%} & \\cellcolor{blue!5}\\textbf{30\\%} & \\cellcolor{blue!5}\\textbf{50\\%} & \\cellcolor{blue!5}\\textbf{Base} & \\cellcolor{blue!5}\\textbf{10\\%} & \\cellcolor{blue!5}\\textbf{30\\%} & \\cellcolor{blue!5}\\textbf{50\\%} \\\\\n\\hline\n'
    for mod in ['RNN', 'CodeBERT', 'AE', 'Qwen']:
        for lang in ['CSharp', 'Java']:
            row = f'\\cellcolor{{blue!5}}{mod} & \\cellcolor{{blue!5}}{lang} '
            for smell in ['FE', 'MA']:
                c_row = agg_clean[(agg_clean['Model'] == mod) & (agg_clean['Lang'] == lang) & (agg_clean['Smell'] == smell)]
                base = c_row['F1' if is_f1 else 'AUC'].values[0]
                row += f'& \\cellcolor{{blue!5}}{base:.3f} '
                for ratio in ['10%', '30%', '50%']:
                    a_row = agg_adv[(agg_adv['Model'] == mod) & (agg_adv['Lang'] == lang) & (agg_adv['Smell'] == smell) & (agg_adv['Ratio'] == ratio)]
                    adv_val = a_row['F1' if is_f1 else 'AUC'].values[0]
                    arrow = get_arrow(adv_val, base)
                    if arrow == '-':
                        row += f'& \\cellcolor{{blue!5}}{adv_val:.3f} - '
                    else:
                        row += f'& \\cellcolor{{blue!5}}{adv_val:.3f} {arrow} '
            row += '\\\\\n'
            latex += row
        latex += '\\hline\n'
    latex += '\\end{tabular}}\n\\end{table}\n'
    return latex
print(build_rq3_latex('F1', True).strip())
print()
print(build_rq3_latex('AUC', False).strip())