import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('./v100_result.csv', sep=None) 

X = df.drop(columns=['CP size','EXP ID','dtype','ffn hidden size','hidden size','vocab size','Peak GPU memory','TFLOP/s/GPU','elapsed time per iteration'])
y = df[['Peak GPU memory','TFLOP/s/GPU','elapsed time per iteration']]

cat_cols = X.select_dtypes(include=['object','int64']).columns.tolist()
num_cols = X.select_dtypes(include=['float64']).columns.tolist()

pre = ColumnTransformer([
        ('cat', 'passthrough', cat_cols),   
        ('num', StandardScaler(), num_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    base = Pipeline(steps=[
                ('prep', pre),
                ('model', MultiOutputRegressor(
                    RandomForestRegressor(n_estimators=800,
                                        max_depth=None,
                                        min_samples_leaf=1,
                                        n_jobs=-1,
                                        random_state=42)))
            ])
    base.fit(X_train, y_train)
    pred = base.predict(X_test)

    for i, t in enumerate(['Memory','TFLOPS','Time']):
        print(f'{t}  MAE={mean_absolute_error(y_test.iloc[:,i], pred[:,i]):.2f}  R2={r2_score(y_test.iloc[:,i], pred[:,i]):.3f}')

    joblib.dump(base, 'rf.pkl')

def test():
    rf_loaded = joblib.load('rf.pkl')

    ext_df = X_test.copy()
    ext_df['num layers'] *= 2

    pred_ext   = rf_loaded.predict(ext_df)[:, 0]     # 外推
    pred_norm  = rf_loaded.predict(X_test)[:, 0]     # 原域对照

    # 4. 外推能力指标
    mae_ext  = mean_absolute_error(y_test.iloc[:,0], pred_ext)
    mae_norm = mean_absolute_error(y_test.iloc[:,0], pred_norm)
    print('显存-外推 MAE :', mae_ext)
    print('显存-原域 MAE :', mae_norm)
    print('外推/原域比值 :', mae_ext / mae_norm)

if __name__ == '__main__':
    test()