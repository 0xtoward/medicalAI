import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# ============================
# 1. 数据加载与清洗 (通用版)
# ============================
def load_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, header=None, engine='openpyxl')
    else:
        df = pd.read_csv(file_path, header=None, encoding='utf-8')

    data = df.iloc[2:].copy()
    
    # 映射列名
    column_map = {
        0: 'RandomID', 4: 'Age', 6: 'Weight', 9: 'Thyroid_Weight', 
        10: 'Ultrasound_Text', 14: 'Outcome', 
        16: 'FT3', 17: 'FT4', 18: 'TSH', 21: 'TRAb',
        22: 'Iodine_Uptake_24h', 25: 'Dose_mCi'
    }
    data = data.rename(columns=column_map)[list(column_map.values())]
    
    # 清洗数值列
    cols = ['Age', 'Weight', 'Thyroid_Weight', 'FT3', 'FT4', 'TSH', 'TRAb', 'Iodine_Uptake_24h', 'Dose_mCi', 'Outcome']
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    
    data = data.dropna(subset=['Outcome']) # 删掉没结局的
    
    # 处理超声文本
    def map_us(t):
        t = str(t)
        if '极' in t or '异常' in t: return 3
        if '增多' in t or '丰富' in t: return 2
        if '未见' in t: return 0
        return 1
    data['Ultrasound_Score'] = data['Ultrasound_Text'].apply(map_us)
    
    return data

# ============================
# 2. 强力特征工程
# ============================
def get_features(df, binary=True):
    # 1. 剂量强度: 每克甲状腺给了多少药?
    df['Dose_Intensity'] = df['Dose_mCi'] / df['Thyroid_Weight']
    # 2. 抗体密度: 每克甲状腺有多少抗体?
    df['TRAb_Density'] = df['TRAb'] / df['Thyroid_Weight']
    # 3. 吸收量估计: 剂量 * 吸碘率 (实际吃到肚子里的辐射)
    df['Effective_Dose'] = df['Dose_mCi'] * (df['Iodine_Uptake_24h'] / 100)
    
    features = ['Age', 'Weight', 'Thyroid_Weight', 'Ultrasound_Score', 
                'FT3', 'TRAb', 'Iodine_Uptake_24h', 'Dose_mCi',
                'Dose_Intensity', 'TRAb_Density', 'Effective_Dose']
    
    X = df[features]
    # 填补缺失值 (用中位数)
    X = X.fillna(X.median())
    
    if binary:
        # --- 二分类：3=治愈(1), 1/2=失败(0) ---
        y = (df['Outcome'] == 3).astype(int)
    else:
        # --- 三分类：保持原始Outcome (1,2,3) ---
        y = df['Outcome'].astype(int)
    
    return X, y, features

# ============================
# 4. 运行随机森林 (二分类 + 过拟合测试)
# ============================
def run_rf_model(file_path, overfit_test=True):
    """
    overfit_test=True: 全量数据训练+预测，测试模型极限能力
    overfit_test=False: 正常5折交叉验证
    """
    print("=" * 70)
    print("🧪 RF 过拟合测试：二分类 + 全量数据训练" if overfit_test else "RF 正常模式：5折交叉验证")
    print("=" * 70)
    
    print("\n正在加载数据...")
    df = load_data(file_path)
    X, y, feat_names = get_features(df, binary=True)  # 使用二分类
    
    print(f"\n样本总数: {len(X)}")
    print(f"失败 (类别0): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    print(f"治愈 (类别1): {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    
    if overfit_test:
        # 过拟合测试：用最强参数，看模型能否完全拟合数据
        print("\n⚠️  过拟合测试模式：训练集=测试集")
        print("目的：测试模型的极限能力，如果准确率仍然低，说明数据本身有问题")
        
        rf = RandomForestClassifier(
            n_estimators=500,      # 更多树
            max_depth=None,        # 不限制深度（允许完全过拟合）
            min_samples_split=2,   # 最小分裂
            min_samples_leaf=1,    # 最小叶子节点
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        print("\n正在训练模型（全量数据）...")
        rf.fit(X, y)
        
        print("正在预测（全量数据）...")
        y_pred = rf.predict(X)
        
        print("\n--- 过拟合测试结果 ---")
        print(f"⚠️  注意：这是在训练集上预测，理论上应该接近100%")
        
    else:
        # 正常5折交叉验证
        print("\n使用默认参数")
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=5, 
            class_weight='balanced', 
            random_state=42
        )
        
        print("\n正在进行 5折交叉验证...")
        y_pred = cross_val_predict(rf, X, y, cv=5)
        
        print("\n--- 交叉验证结果 ---")
    
    # 评估
    acc = accuracy_score(y, y_pred)
    print(f"\n准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("\n详细分类报告:")
    print(classification_report(y, y_pred, target_names=['失败(Fail)', '治愈(Cure)']))
    
    # 特征重要性
    if not overfit_test:
        rf.fit(X, y)
    
    importances = pd.DataFrame({'Feature': feat_names, 'Importance': rf.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 5 关键特征 ---")
    print(importances.head(5).to_string(index=False))
    
    # 诊断结论
    if overfit_test:
        print("\n" + "=" * 70)
        print("🔍 诊断结论:")
        if acc < 0.7:
            print("❌ 即使过拟合，准确率仍<70% → 数据质量存在根本性问题！")
            print("   可能原因：")
            print("   1. 标注错误（Outcome标签不准确）")
            print("   2. 关键特征缺失（影响结果的因素没有被记录）")
            print("   3. 数据本身噪音过大（随机性强）")
        elif acc < 0.85:
            print("⚠️  过拟合准确率70-85% → 数据有一定模式，但噪音较大")
            print("   建议：数据清洗、增加特征、检查异常值")
        else:
            print("✅ 过拟合准确率>85% → 模型能力OK，需要更多数据或正则化")
        print("=" * 70)

if __name__ == "__main__":
    file_name = "700.xlsx"
    
    # 🔬 诊断模式：测试模型能否完全拟合数据
    # overfit_test=True: 全量数据训练+预测（测试数据质量）
    # overfit_test=False: 正常5折交叉验证
    
    try:
        print("\n" + "🔬 " * 35)
        print("诊断测试：检查数据是否可学习")
        print("🔬 " * 35 + "\n")
        
        # 运行过拟合测试
        run_rf_model(file_name, overfit_test=True)
        
        print("\n\n" + "📊 " * 35)
        print("对比：正常交叉验证结果")
        print("📊 " * 35 + "\n")
        
        # 运行正常验证（对比）
        run_rf_model(file_name, overfit_test=False)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")