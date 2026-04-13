import pandas as pd
import numpy as np

def main():
    try:
        df_clean = pd.read_csv('../results/rq1_results.csv')
        df_adv = pd.read_csv('../results/rq2_results.csv')
    except FileNotFoundError:
        print("Could not find results files. Please run from table_generation directory.")
        return

    agg_clean = df_clean.groupby(['Model', 'Lang', 'Smell']).mean(numeric_only=True).reset_index()
    agg_adv = df_adv.groupby(['Model', 'Lang', 'Smell', 'Mode']).mean(numeric_only=True).reset_index()

    models = ['RNN', 'CodeBERT', 'AE', 'Qwen']
    langs = ['CSharp', 'Java']
    modes = ['Semantic', 'Structural', 'Hybrid', 'NLP']
    smells = ['CM', 'CC', 'FE', 'MA']

    def print_table(metric):
        print(f"\nRQ2: Model {metric} Degradation on Adversarial Test Samples")
        print("-" * 120)
        header = f"{'Model':<10} | {'Lang':<8} | {'Attack':<12} | "
        header += " | ".join([f"{s + ' (Cln->Adv)':<17}" for s in smells])
        print(header)
        print("-" * 120)

        for model in models:
            for lang in langs:
                for mode in modes:
                    row_str = f"{model:<10} | {lang:<8} | {mode:<12} | "
                    cells = []
                    for smell in smells:
                        c_row = agg_clean[(agg_clean['Model'] == model) & (agg_clean['Lang'] == lang) & (agg_clean['Smell'] == smell)]
                        a_row = agg_adv[(agg_adv['Model'] == model) & (agg_adv['Lang'] == lang) & (agg_adv['Smell'] == smell) & (agg_adv['Mode'] == mode)]
                        
                        if len(c_row) > 0 and len(a_row) > 0:
                            c_val = c_row[metric].values[0]
                            a_val = a_row[metric].values[0]
                            
                            arrow = '-'
                            if round(a_val, 3) < round(c_val, 3): arrow = 'v'
                            elif round(a_val, 3) > round(c_val, 3): arrow = '^'
                            
                            cells.append(f"{c_val:.3f}->{a_val:.3f} {arrow}")
                        else:
                            cells.append(f"{'-':<17}")
                    row_str += " | ".join([f"{c:<17}" for c in cells])
                    print(row_str)
            print("-" * 120)

    print_table('F1')
    print_table('AUC')

if __name__ == '__main__':
    main()
