import pandas as pd
import numpy as np
import os
try:
    import nlpaug.augmenter.word as naw
except ImportError:
    naw = None

class ASAGDataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if naw:
            try:
                self.augmenter = naw.SynonymAug(aug_src='wordnet')
            except:
                self.augmenter = None
        else:
            self.augmenter = None
        
    def load_local_dataset(self, filename, student_col, ref_col, score_col, question_col=None):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return pd.DataFrame()
            
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".tsv"):
                df = pd.read_csv(path, sep='\t')
            else:
                df = pd.read_excel(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return pd.DataFrame()
            
        # Standardize columns
        columns_to_keep = [student_col, ref_col, score_col]
        if question_col and question_col in df.columns:
            columns_to_keep.append(question_col)
            
        df = df.dropna(subset=columns_to_keep).copy()
        
        df.rename(columns={
            student_col: 'student_answer',
            ref_col: 'reference_answer',
            score_col: 'score'
        }, inplace=True)
        
        if question_col and question_col in df.columns:
            df.rename(columns={question_col: 'question'}, inplace=True)
        else:
            df['question'] = ""
            
        # Normalize score to 0-1
        min_score = df['score'].min()
        max_score = df['score'].max()
        if max_score > min_score:
            df['normalized_score'] = (df['score'] - min_score) / (max_score - min_score)
        else:
            df['normalized_score'] = 0.5
            
        return df

    def get_unified_dataset(self):
        """
        Loads and combines all available ASAG datasets.
        """
        datasets = []
        
        # 1. Mohler_Mihalcea_dataset (CS context)
        df_mohler = self.load_local_dataset(
            "Mohler_Mihalcea_dataset.csv", 
            student_col="student_answer", 
            ref_col="instructor_answer", 
            score_col="score_avg",
            question_col="question"
        )
        if not df_mohler.empty:
            datasets.append(df_mohler)
            
        # 2. scientsbank_train (Science context)
        df_scientsbank = self.load_local_dataset(
            "scientsbank_train.csv",
            student_col="studentAnswer",
            ref_col="referenceAnswer",
            score_col="score"
        )
        if not df_scientsbank.empty:
            datasets.append(df_scientsbank)
            
        # Combine
        if datasets:
            return pd.concat(datasets, ignore_index=True)
        return pd.DataFrame()
        
    def augment_data(self, text):
        """Applies synonym replacement for robustness."""
        if self.augmenter and pd.notna(text):
            try:
                # nlpaug can return a list or string, ensure string
                aug = self.augmenter.augment(str(text))
                return aug[0] if isinstance(aug, list) else aug
            except:
                return text
        return text
