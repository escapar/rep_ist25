import os
import pandas as pd
import glob
CS_METRICS_BASE = 'data/designite_out_cs_extracted/designite_out'
JAVA_METRICS_BASE = 'data/designite_out_java_extracted/designite_out_java'

class DesigniteMetricLoader:

    def __init__(self, lang):
        self.lang = lang
        self.base_path = CS_METRICS_BASE if lang == 'CSharp' else JAVA_METRICS_BASE
        self.cache = {}

    def _get_cs_metrics(self, project_name, metric_type):
        key = (project_name, metric_type)
        if key in self.cache:
            return self.cache[key]
        proj_dir = os.path.join(self.base_path, project_name)
        if not os.path.exists(proj_dir):
            return None
        suffix = '_MethodMetrics.csv' if metric_type == 'Method' else '_ClassMetrics.csv'
        pattern = os.path.join(proj_dir, f'*{suffix}')
        csv_files = glob.glob(pattern)
        if not csv_files:
            return None
        dfs = []
        for f in csv_files:
            try:
                dfs.append(pd.read_csv(f))
            except:
                pass
        if not dfs:
            return None
        combined = pd.concat(dfs)
        self.cache[key] = combined
        return combined

    def _get_java_metrics(self, project_name, metric_type):
        key = (project_name, metric_type)
        if key in self.cache:
            return self.cache[key]
        proj_dir = os.path.join(self.base_path, project_name)
        if not os.path.exists(proj_dir):
            return None
        csv_file = 'MethodMetrics.csv' if metric_type == 'Method' else 'TypeMetrics.csv'
        path = os.path.join(proj_dir, csv_file)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            self.cache[key] = df
            return df
        except:
            return None

    def get_metrics_for_file(self, filename, smell):
        parts = filename.replace('.code', '').replace('.java', '').split('_')
        if len(parts) < 3:
            return None
        project = parts[0]
        metric_type = 'Method' if smell in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy'] else 'Class'
        if self.lang == 'CSharp':
            df = self._get_cs_metrics(project, metric_type)
            if df is None:
                return None
            target = parts[-1]
            if metric_type == 'Method':
                match = df[df['Method Name'] == target]
                if not match.empty:
                    row = match.iloc[0]
                    return [row.get('LOC', 0), row.get('CC', 0), row.get('PC', 0)]
            else:
                match = df[df['Class Name'] == target]
                if not match.empty:
                    row = match.iloc[0]
                    return [row.get('LOC', 0), row.get('NOM', 0), row.get('WMC', 0)]
        else:
            df = self._get_java_metrics(project, metric_type)
            if df is None:
                return None
            target = parts[-1]
            if metric_type == 'Method':
                match = df[df['Method Name'] == target]
                if not match.empty:
                    row = match.iloc[0]
                    return [row.get('LOC', 0), row.get('CC', 0), row.get('PC', 0)]
            else:
                match = df[df['Type Name'] == target]
                if not match.empty:
                    row = match.iloc[0]
                    return [row.get('LOC', 0), row.get('NOM', 0), row.get('WMC', 0), row.get('FANIN', 0)]
        return None
if __name__ == '__main__':
    loader = DesigniteMetricLoader('Java')
    print(loader.get_metrics_for_file('247687009_soa_..._someMethod.java', 'ComplexMethod'))