import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

# ==========================================
# 1. 专门针对你数据的清洗与加载函数
# ==========================================
def load_and_clean_data(file_path):
    # --- 修复点：自动判断是 Excel 还是 CSV ---
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        # 即使没有表头，也加上 header=None，防止第一行数据被误吞
        df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
    else:
        # 如果是 CSV，尝试不同的编码，防止中文乱码
        try:
            df_raw = pd.read_csv(file_path, header=None, encoding='utf-8')
        except UnicodeDecodeError:
            df_raw = pd.read_csv(file_path, header=None, encoding='gbk')
            
    # 数据从第3行（索引2）开始
    data = df_raw.iloc[2:].copy()
    
    # ... (后续代码保持不变)
    
    # 手动重命名关键列（根据你的Excel结构）
    column_map = {
        0: 'RandomID', 1: 'HospID', 2: 'Name', 3: 'Sex', 4: 'Age', 
        5: 'Height', 6: 'Weight', 7: 'BMI', 9: 'Thyroid_Weight', 
        10: 'Ultrasound_Text', 
        12: 'Treatment_Count', 14: 'Outcome', 
        16: 'FT3', 17: 'FT4', 18: 'TSH', 
        19: 'TGAb', 20: 'TPOAb', 21: 'TRAb',
        22: 'Iodine_Uptake_24h', 25: 'Dose_mCi'
    }
    data = data.rename(columns=column_map)
    data = data[list(column_map.values())] 

    # --- 特征工程 A: 处理“超声”文本列 ---
    def map_ultrasound(text):
        if pd.isna(text): return np.nan
        text = str(text)
        if any(x in text for x in ['极丰富', '异常', '丰富']): return 3
        if any(x in text for x in ['较丰富', '增多']): return 2
        if any(x in text for x in ['略', '稍']): return 1
        if '未见' in text: return 0
        return 1 
    
    data['Ultrasound_Score'] = data['Ultrasound_Text'].apply(map_ultrasound)

    # --- 特征工程 B: 数值列清洗 ---
    numeric_cols = ['Age', 'Height', 'Weight', 'BMI', 'Thyroid_Weight', 
                    'FT3', 'FT4', 'TSH', 'TGAb', 'TPOAb', 'TRAb', 
                    'Iodine_Uptake_24h', 'Dose_mCi', 'Outcome']
    
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=['Outcome'])
    
    return data
# ==========================================
# 2. 高级特征构造 (Feature Engineering)
# ==========================================
def engineer_features(df):
    # 构造医学交互特征（与rf.py保持一致）
    # 剂量/重量比 (mCi/g)：这是比单纯剂量更重要的指标
    df['Dose_Intensity'] = df['Dose_mCi'] / df['Thyroid_Weight']
    
    # 抗体负荷 (TRAb/Weight)：也许小甲状腺高抗体更难治？
    df['TRAb_Density'] = df['TRAb'] / df['Thyroid_Weight']
    
    # 吸收量估计: 剂量 * 吸碘率 (实际吃到肚子里的辐射)
    df['Effective_Dose'] = df['Dose_mCi'] * (df['Iodine_Uptake_24h'] / 100)
    
    # 选择最终入模特征（与rf.py一致）
    feature_cols = [
        'Age', 'Weight', 'Thyroid_Weight', 'Ultrasound_Score', 
        'FT3', 'TRAb', 'Iodine_Uptake_24h', 'Dose_mCi',
        'Dose_Intensity', 'TRAb_Density', 'Effective_Dose'
    ]
    
    X = df[feature_cols]
    
    # --- 改为三分类：保持原始Outcome (1,2,3) ---
    y = df['Outcome'].astype(int)
    
    return X, y, feature_cols

# ==========================================
# 3. 运行 TabPFN 模型
# ==========================================
def run_tabpfn_pipeline(file_path):
    print("正在加载数据...")
    df = load_and_clean_data(file_path)
    X, y, feat_names = engineer_features(df)
    
    print(f"样本总数: {len(X)}")
    print(f"类别1(甲亢)数量: {sum(y==1)}")
    print(f"类别2(甲减)数量: {sum(y==2)}")
    print(f"类别3(正常)数量: {sum(y==3)}")
    
    # 填补缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        print(f"错误: TabPFN导入失败: {e}")
        print("请尝试: pip install tabpfn")
        raise

    # --- 修正点：移除旧版参数，保留 device='cpu' 即可 ---
    # 如果你也遇到 device 参数报错，可以连 device='cpu' 也删掉，直接写 TabPFNClassifier()
    classifier = TabPFNClassifier(device='cpu') 
    
    print("正在进行 In-Context Learning (预测)...")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    # 评估
    print("\n--- 最终成绩单 ---")
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred))
    
    return classifier, X_test, y_test
# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 替换成你的文件名
    file_name = "700.xlsx" 
    
    try:
        model, X_test, y_test = run_tabpfn_pipeline(file_name)
        print("\n✅ 工作流运行成功！")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")