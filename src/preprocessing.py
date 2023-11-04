from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

class DataSplitter:

    def __init__(self):
        self._preprocessor = None

    @staticmethod
    def split_data(df, num_features, cat_features, target_column):

        df_train, df_valid = train_test_split(
            df[num_features + cat_features + [target_column]], test_size=0.25, 
            stratify=df[target_column], shuffle=True, random_state=42
        )

        return df_train, df_valid

class DataTransformer:

    def __init__(self, num_features, cat_features):
        self.num_features = num_features
        self.cat_features = cat_features
        self._preprocessor = None

    def fit(self, X):
        self._preprocessor = ColumnTransformer([
            ("num_preprocess", StandardScaler(), self.num_features),
            ("cat_preprocess", OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), self.cat_features)
        ])

        self._preprocessor.fit(X)
        return self

    def transform(self, X):
        return self._preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def add_custom_features(df):
        df['active_contact'] = (df['previous'] != 0).astype(int)
        df['without_debt'] = ((df['default'] == 'no') & (df['housing'] == 'no') & (df['loan'] == 'no')).astype(int)
        return df