import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('../results/rq1_results.csv')
    except FileNotFoundError:
        print("Could not find ../results/rq1_results.csv. Please run from table_generation directory.")
        return

    agg = df.groupby(['Model', 'Lang', 'Smell']).mean(numeric_only=True).reset_index()

    print("RQ1: Baseline Model Performance on Clean Test Data (F1 / AUC)")
    print("-" * 80)
    
    models = ['RNN', 'CodeBERT', 'AE', 'Qwen']
    langs = ['CSharp', 'Java']
    smells = ['CM', 'CC', 'FE', 'MA']

    header = f"{'Model':<12} | {'Lang':<10} | " + " | ".join([f"{s:<13}" for s in smells])
    print(header)
    print("-" * 80)

    for model in models:
        for lang in langs:
            row_str = f"{model:<12} | {lang:<10} | "
            cells = []
            for smell in smells:
                row_data = agg[(agg['Model'] == model) & (agg['Lang'] == lang) & (agg['Smell'] == smell)]
                if len(row_data) > 0:
                    f1 = row_data['F1'].values[0]
                    auc = row_data['AUC'].values[0]
                    cells.append(f"{f1:.3f} / {auc:.3f}")
                else:
                    cells.append(f"{'-':<13}")
            row_str += " | ".join([f"{c:<13}" for c in cells])
            print(row_str)
        print("-" * 80)

if __name__ == '__main__':
    main()
