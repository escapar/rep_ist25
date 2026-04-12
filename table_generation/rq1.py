import pandas as pd
df = pd.read_csv('../data_csvs/rq1_results.csv')
agg = df.groupby(['Model', 'Lang', 'Smell']).mean().reset_index()
latex = '\\begin{table}[H]\n\\caption{\\blue{Baseline Model Performance on Clean Test Data (F1 / AUC)}}\n\\label{table_rq1_new}\n\\centering\\footnotesize\n\\begin{tabular}{llcccc}\n\\hline\n\\cellcolor{blue!5} \\textbf{Model} & \\cellcolor{blue!5} \\textbf{Lang} & \\cellcolor{blue!5} \\textbf{CM} & \\cellcolor{blue!5} \\textbf{CC} & \\cellcolor{blue!5} \\textbf{FE} & \\cellcolor{blue!5} \\textbf{MA} \\\\\n\\hline\n'
for model in ['RNN', 'CodeBERT', 'AE', 'Qwen']:
    for lang in ['CSharp', 'Java']:
        cell_m = '\\cellcolor{blue!5} ' if model != 'RNN' else '\\cellcolor{blue!5} '
        row = f'{cell_m}{model} & \\cellcolor{{blue!5}} {lang}'
        for smell in ['CM', 'CC', 'FE', 'MA']:
            row_data = agg[(agg['Model'] == model) & (agg['Lang'] == lang) & (agg['Smell'] == smell)]
            f1 = row_data['F1'].values[0]
            auc = row_data['AUC'].values[0]
            row += f' & \\cellcolor{{blue!5}} {f1:.3f} / {auc:.3f}'
        row += ' \\\\\n'
        latex += row
    latex += '\\hline\n'
latex += '\\end{tabular}\n\\end{table}\n'
print(latex)