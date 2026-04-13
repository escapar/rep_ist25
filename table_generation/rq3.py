import pandas as pd
import numpy as np

def main():
    try:
        df_clean_base = pd.read_csv('../results/rq1_results.csv')
        df_clean_advtrain = pd.read_csv('../results/rq3_clean_results.csv')
        df_adv_advtrain = pd.read_csv('../results/rq3_adv_results.csv')
        
        # RQ2 results to get adversarial base
        df_adv_base = pd.read_csv('../results/rq2_results.csv')
    except FileNotFoundError:
        print("Could not find results files. Please run from table_generation directory.")
        return

    agg_clean_base = df_clean_base.groupby(['Model', 'Lang', 'Smell']).mean(numeric_only=True).reset_index()
    agg_clean_advtrain = df_clean_advtrain.groupby(['Model', 'Lang', 'Smell', 'Ratio']).mean(numeric_only=True).reset_index()
    agg_adv_advtrain = df_adv_advtrain.groupby(['Model', 'Lang', 'Smell', 'Ratio']).mean(numeric_only=True).reset_index()
    agg_adv_base = df_adv_base.groupby(['Model', 'Lang', 'Smell']).mean(numeric_only=True).reset_index()

    models = ['RNN', 'CodeBERT', 'AE', 'Qwen']
    langs = ['CSharp', 'Java']
    ratios = ['10%', '30%', '50%']

    def print_table(test_data_type, metric, base_agg, advtrain_agg):
        print(f"\nRQ3: {metric} on {test_data_type} Test Data After Adversarial Training")
        print("-" * 140)
        
        for smell_group in [('CM', 'CC'), ('FE', 'MA')]:
            header = f"{'Model':<10} | {'Lang':<8} | "
            for s in smell_group:
                header += f" {s} Base |  {s} 10% |  {s} 30% |  {s} 50% | "
            print(header)
            print("-" * 140)
            
            for model in models:
                for lang in langs:
                    row_str = f"{model:<10} | {lang:<8} | "
                    cells = []
                    for smell in smell_group:
                        if test_data_type == 'Clean':
                            b_row = base_agg[(base_agg['Model'] == model) & (base_agg['Lang'] == lang) & (base_agg['Smell'] == smell)]
                        else:
                            b_row = base_agg[(base_agg['Model'] == model) & (base_agg['Lang'] == lang) & (base_agg['Smell'] == smell)]
                            
                        if len(b_row) > 0:
                            b_val = b_row[metric].values[0]
                            cells.append(f"{b_val:.3f}")
                            
                            for ratio in ratios:
                                a_row = advtrain_agg[(advtrain_agg['Model'] == model) & (advtrain_agg['Lang'] == lang) & (advtrain_agg['Smell'] == smell) & (advtrain_agg['Ratio'] == ratio)]
                                if len(a_row) > 0:
                                    a_val = a_row[metric].values[0]
                                    arrow = '-'
                                    if round(a_val, 3) < round(b_val, 3): arrow = 'v'
                                    elif round(a_val, 3) > round(b_val, 3): arrow = '^'
                                    if (metric == 'F1' and b_val == 0.0) or (metric == 'AUC' and b_val == 0.5):
                                        arrow = '-'
                                    cells.append(f"{a_val:.3f} {arrow}")
                                else:
                                    cells.append(f"{'-':<7}")
                        else:
                            cells.extend([f"{'-':<7}"] * 4)
                            
                    row_str += " | ".join([f"{c:<7}" for c in cells])
                    print(row_str)
            print("-" * 140)

    print_table('Clean', 'F1', agg_clean_base, agg_clean_advtrain)
    print_table('Clean', 'AUC', agg_clean_base, agg_clean_advtrain)
    
    print_table('Adversarial', 'F1', agg_adv_base, agg_adv_advtrain)
    print_table('Adversarial', 'AUC', agg_adv_base, agg_adv_advtrain)

if __name__ == '__main__':
    main()
