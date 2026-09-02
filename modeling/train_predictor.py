import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
import math

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./result_recompute_19.csv', na_values=None, keep_default_na=False).dropna(axis=1, how='all')
#筛除一些之前自定义的列
# df = df.drop(columns=['amp', 'error', 'Unnamed: 25'])

df['dtype_hardware_gen'] = df['dtype'].map({
    'fp16': 1, 
    'bf16': 2 
})

df['dtype_needs_loss_scaling'] = df['dtype'].map({
    'fp16': 1,   
    'bf16': 0    
})

# 在X中新增以下特征列
df['param_memory'] = (
    2 * df['vocab size'] * df['hidden size'] +  # 嵌入层
    df['num layers'] * (
        4 * df['hidden size'] * df['hidden size'] +  # QKV+O投影
        2 * df['hidden size'] * df['ffn hidden size']  # FFN两层
    ) * 2  # fp16/bf16=2字节
)

df['activation_per_layer'] = (
    df['micro batch size'] * df['sequence length'] * 
    df['hidden size'] * 34  # 典型Transformer每层激活系数
)

df['tp_memory_reduction'] = 1 / df['TP size']  # TP切分降低单卡显存

X = df.drop(columns=['EXP ID','dtype','ffn hidden size','hidden size','vocab size','Peak GPU memory','TFLOP/s/GPU','elapsed time per iteration'])
# y = df[['Peak GPU memory','TFLOP/s/GPU','elapsed time per iteration']]
y_mem = df[['Peak GPU memory']]
y_time = df[['elapsed time per iteration']]
y_tflops = df[['TFLOP/s/GPU']]

cat_cols = ['Rank','device type','recompute granularity','dtype_hardware_gen','dtype_needs_loss_scaling']
num_cols = [col for col in X.columns if col not in cat_cols]    

def data_aug(X, y, target_col='elapsed time per iteration'):
    EXT_COLS = ['micro batch size','num layers','sequence length']
    max_value_ext = [16, 4, 4096] #对应ext
    min_value_ext = [1, 1, 1024] #对应ext
    # 线性插值假设（§3.2.2）对时间和显存分别成立，调用方指定目标列
    TARGET_COLS = [target_col]

    coef_dict = {}

    orig_idx = X.index.tolist()

    orig_length = len(orig_idx)
    print("original length:",orig_length)

    experiment_dict = set() 
    single = set()
    num_error = {}
    num_ext={}
    X1,y1 = X,y
    for idx, ext_col in enumerate(EXT_COLS):
        # print(ext_col)
        num_error[ext_col]=0
        num_ext[ext_col]=0
        BACK_COLS = [c for c in X.columns if c != ext_col]
        # print(BACK_COLS)
        for bg_vals, sub_df in X.groupby(BACK_COLS):
            num_ext[ext_col]+=1
            if len(sub_df) < 2:# 至少 2 条才拟合
                for _, row in sub_df.iterrows():
                    single.add(tuple(row))
                orig_idx = [idx for idx in orig_idx if idx not in sub_df.index]
                continue
            sub = sub_df.nsmallest(2, ext_col)

            for _, row in sub.iterrows():
                experiment_dict.add(tuple(row))

            sub_rest = sub_df.drop(sub.index)

            sub_X = sub[[ext_col]].values.flatten()      
            sub_y = y.loc[sub.index, TARGET_COLS].values.flatten()  

            a = (sub_y[1] - sub_y[0]) / (sub_X[1] - sub_X[0])        
            b = sub_y[0] - a * sub_X[0]

            if a < 0:
                num_error[ext_col]+=1
                print(sub.to_string())
                
                # print("sub_X:",sub_X)
                # print("sub_y:",sub_y)
                print("_____________")                    

            coef_dict[bg_vals] = (np.array([a]), b)

        if len(coef_dict) == 0:
            continue
            
        orig_min = X[[ext_col]].min().values[0]
        orig_max = X[[ext_col]].max().values[0]
        low = min_value_ext[idx]

        high = max_value_ext[idx]
        # print(f"{ext_col} ori_min:{orig_min} ori_max:{orig_max} high:{high}")
        n = int(math.log(high/low, 2))+1
        n_aug = int(len(coef_dict)*(n)) #根据区间范围放大
        # print(f"coef_dict:{len(coef_dict)} n_aug:{n_aug}")
        exist_bg = set(coef_dict.keys())
        orig_data = X[X[BACK_COLS].apply(lambda x: tuple(x) in exist_bg, axis=1)]
        orig_data = orig_data.drop_duplicates(subset=BACK_COLS)
        # 成倍放大原始数据
        aug = pd.concat([orig_data] * (n))  # 重复原始数据
        print(len(aug))
        # aug = aug.head(n_aug)  # 截取到所需数量
        # print(len(aug))
        points_per_segment = n_aug // n
        segments = [low * (2 ** i) for i in range(n) for _ in range(points_per_segment)]
        # print(len(segments))
        # 计算每段的数据点数
    
        ext_val = np.array(segments).reshape(-1, 1)
        # ext_val = np.random.uniform(low, high, size=(n_aug, 1))
        aug[[ext_col]] = ext_val

        y_aug = np.zeros((n_aug,1))
        for i, (idx, row) in enumerate(aug.iterrows()):
            bg_vals = tuple(row[BACK_COLS])
            
            a, b = coef_dict.get(bg_vals)
            a = a.reshape(-1) 
            y_aug[i] = np.dot(ext_val[i, :], a) + b   # y = Ax + b（单目标
            # if y_aug[i] < 0:
            #     num_error[ext_col]+=1
            #     # print("ext_col:",ext_col)
            #     # print("a:",a)
            #     # print("b:",b)
            #     # print("ext_val[i, :]:",ext_val[i, :])
            #     # print("y_aug_mem[i]:",y_aug_mem[i])
            #     # print("_____________")
        y_aug = np.squeeze(y_aug)
    
        y_aug = pd.Series(y_aug, name=TARGET_COLS[0])

        y1 = pd.concat([y1[TARGET_COLS[0]].to_frame(),   # ← 转回 DataFrame
               y_aug.to_frame()], ignore_index=True)
        X1 = pd.concat([X1, aug], ignore_index=True)

        coef_dict.clear()

    print("num_single:",len(single))
    print("num_error:",num_error)

    X_new = X1.iloc[len(orig_idx):]
    y_new = y1.iloc[len(orig_idx):]
    
    return X_new, y_new


def train(X_train, X_test, y_train, y_test, model_name, n_estimators=30000):
    import xgboost as xgb
    import numpy as np

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        print(X_train[col].unique())
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        print(X_train[col].unique())
        label_encoders[col] = le
    
    
    # 2. 模型打开 categorical 支持
    xgb_single = xgb.XGBRegressor(
        n_estimators=n_estimators, # 默认足够大，配合 early stopping
        learning_rate=0.03,
        max_depth=0,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=2.5,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric='mae',
        enable_categorical=True,
        tree_method='hist')

    xgb_single.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=0)
    print("best_iteration:", xgb_single.best_iteration)

    print(xgb_single.feature_importances_)

    # 7. 预测 + 评估
    y_pred = xgb_single.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f'{model_name} MAE={mae:7.2f}  R2={r2:.4f}')

    joblib.dump(xgb_single, model_name+'.pkl')
    joblib.dump(label_encoders, model_name+'_labels.pkl')
    return xgb_single, label_encoders

def test(model,X_test,y_test):
    rf_loaded = joblib.load(model+'.pkl')

    y_pred = rf_loaded.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f'{model} MAE={mae:7.2f}  R2={r2:.4f}')

def test_single(model_name):
    single = {
        'TP size': 1,
        'Rank': 0,
        'device type': 'v100',
        'max position embedding': 2048,
        'micro batch size': 1,
        'num attention heads': 1,
        'num layers': 4,         
        'num query groups': 1, 
        'recompute granularity': 'full-uniform-1',
        'sequence length': 1024,
    }

    single_df = pd.DataFrame([single])

    # for col in cat_cols:          
    #     single_df[col] = single_df[col].astype('category')
    
    rf_loaded = joblib.load(model_name+'.pkl')
    label_encoders = joblib.load(model_name+'_labels.pkl')

    for col in label_encoders.keys():
        le = label_encoders[col]
        single_df[col] = le.transform(single_df[col])
    
    pred = rf_loaded.predict(single_df)   
    print('预测结果:', pred[0])

if __name__ == '__main__':
    from dual_predictor import DualOutputPredictor

    # ---- 时间模型：线性插值数据增强后训练 ----
    new_X, new_y_time = data_aug(X, y_time, target_col='elapsed time per iteration')
    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y_time, test_size=0.2, random_state=42)
    time_model, enc_time = train(X_train, X_test, y_train, y_test, 'xgb_time_aug')

    # ---- 显存模型：同样的增强流程，目标列换成 Peak GPU memory ----
    new_Xm, new_y_mem = data_aug(X, y_mem, target_col='Peak GPU memory')
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(new_Xm, new_y_mem, test_size=0.2, random_state=42)
    mem_model, enc_mem = train(Xm_train, Xm_test, ym_train, ym_test, 'xgb_mem_aug')

    # ---- 打包为双输出预测器：搜索阶段一次推理同时得到时间和显存（§3.2.2）----
    # 两个模型的 LabelEncoder 拟合自同一特征空间，类别集合一致，可共用
    dual = DualOutputPredictor(time_model, mem_model, enc_time)
    dual.save('xgb_dual_aug.pkl')
    print('saved xgb_dual_aug.pkl (time + memory dual-output predictor)')