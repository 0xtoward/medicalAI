import pandas as pd
import numpy as np
from collections import Counter

FILE_PATH = "1003.xlsx"
EVAL_COLS = [35, 44, 53, 62, 71]  # 1M, 3M, 6M, 1Y, 1.5Y
TIME_LABELS = ["1M", "3M", "6M", "1Y", "1.5Y"]
STATUS = {1: "甲亢", 2: "甲减", 3: "正常"}

def load_all():
    """加载全部数据，返回首次治疗df和多次治疗列表"""
    df_raw = pd.read_excel(FILE_PATH, header=None, engine='openpyxl').iloc[2:]
    outcome_raw = pd.to_numeric(df_raw.iloc[:, 14], errors='coerce')
    df_raw = df_raw.loc[outcome_raw.notna()].reset_index(drop=True)

    first_rows = []
    retreatment = []
    i = 0
    while i < len(df_raw):
        if pd.notna(df_raw.iloc[i, 0]):
            first_rows.append(i)
            treatments = [int(pd.to_numeric(df_raw.iloc[i, 14], errors='coerce'))]
            doses = [df_raw.iloc[i, 25]]
            j = i + 1
            while j < len(df_raw) and pd.isna(df_raw.iloc[j, 0]):
                treatments.append(int(pd.to_numeric(df_raw.iloc[j, 14], errors='coerce')))
                doses.append(df_raw.iloc[j, 25])
                j += 1
            if len(treatments) > 1:
                retreatment.append({
                    'id': df_raw.iloc[i, 0],
                    'age': df_raw.iloc[i, 4],
                    'outcomes': treatments,
                    'doses': doses,
                })
            i = j
        else:
            i += 1

    df = df_raw.iloc[first_rows].reset_index(drop=True)
    return df, retreatment


def get_trajectory(eval_seq_row):
    """提取有效的随访序列"""
    return [(TIME_LABELS[j], int(eval_seq_row.iloc[j]))
            for j in range(len(TIME_LABELS))
            if not np.isnan(eval_seq_row.iloc[j])]


def classify(seq, final):
    """
    互斥分类 (按最终结局 + 轨迹)

    最终甲亢:
      A1 持续甲亢    全程1，从未好转
      A2 复发甲亢    中间出现过2或3，但最终回到1

    最终正常:
      B1 持续正常    全程3
      B2 好转正常    从1逐步好转到3，无回头
      B3 波动后正常  中间有反复(出现过1→3→1等)最终3

    最终甲减:
      C1 迟发甲减    出现过3(正常)之后变2
      C2 直接甲减    从未出现过3，1→2
      C3 波动后甲减  其他路径到甲减
    """
    statuses = [s for _, s in seq]
    if len(statuses) == 0:
        return "X_no_data"

    if final == 1:
        if all(s == 1 for s in statuses):
            return "A1_persistent_hyper"
        else:
            return "A2_relapsed_hyper"

    elif final == 3:
        if all(s == 3 for s in statuses):
            return "B1_stable_normal"
        # 单调好转: 没有从非1回到1的情况
        saw_non_hyper = False
        monotonic = True
        for s in statuses:
            if s in [2, 3]:
                saw_non_hyper = True
            elif saw_non_hyper and s == 1:
                monotonic = False
                break
        if monotonic:
            return "B2_improving_normal"
        else:
            return "B3_fluctuating_normal"

    elif final == 2:
        if 3 in statuses:
            return "C1_late_hypo"
        elif 3 not in statuses:
            return "C2_direct_hypo"
        else:
            return "C3_fluctuating_hypo"

    return "X_unknown"


def main():
    df, retreatment = load_all()
    n = len(df)
    final_outcome = pd.to_numeric(df.iloc[:, 14], errors='coerce').astype(int)

    eval_seq = df.iloc[:, EVAL_COLS].apply(pd.to_numeric, errors='coerce')
    eval_seq.columns = TIME_LABELS

    print(f"{'='*70}")
    print(f"  患者转归轨迹分析")
    print(f"{'='*70}")
    print(f"  首次治疗患者: {n}")
    print(f"  接受多次治疗: {len(retreatment)}")
    print(f"  总治疗记录:   {n + sum(len(r['outcomes'])-1 for r in retreatment)}")

    # ========== 各时间点分布（含比例） ==========
    print(f"\n  各时间点状态分布:")
    print(f"  {'时间':>5}  {'甲亢':>6} {'(%)':>7}  {'甲减':>6} {'(%)':>7}  {'正常':>6} {'(%)':>7}  {'缺失':>5}")
    print(f"  {'─'*62}")
    for t in TIME_LABELS:
        col = eval_seq[t]
        n_valid = col.notna().sum()
        n_hyper = (col == 1).sum()
        n_hypo  = (col == 2).sum()
        n_norm  = (col == 3).sum()
        n_miss  = col.isna().sum()
        if n_valid > 0:
            print(f"  {t:>5}  {n_hyper:>6} {n_hyper/n_valid:>6.1%}  "
                  f"{n_hypo:>6} {n_hypo/n_valid:>6.1%}  "
                  f"{n_norm:>6} {n_norm/n_valid:>6.1%}  {n_miss:>5}")
        else:
            print(f"  {t:>5}  {n_hyper:>6}    ---  {n_hypo:>6}    ---  {n_norm:>6}    ---  {n_miss:>5}")
    print(f"  {'─'*62}")
    print(f"  * 百分比按该时间点有数据的患者计算")

    # ========== 最终结局 ==========
    print(f"\n  最终结局(col14):")
    for v in [1, 2, 3]:
        cnt = (final_outcome == v).sum()
        print(f"    {STATUS[v]}: {cnt} ({cnt/n:.1%})")

    # ========== 互斥分类 ==========
    cats = {}
    for i in range(n):
        seq = get_trajectory(eval_seq.iloc[i])
        cats[i] = classify(seq, final_outcome.iloc[i])

    cat_series = pd.Series(cats)

    LABEL_MAP = {
        "A1_persistent_hyper":   ("最终甲亢", "持续甲亢 (全程甲亢，从未好转)"),
        "A2_relapsed_hyper":     ("最终甲亢", "复发甲亢 (中间好转过，又复发)"),
        "B1_stable_normal":      ("最终正常", "持续正常 (全程正常，一步到位)"),
        "B2_improving_normal":   ("最终正常", "好转正常 (逐步好转，无反复)"),
        "B3_fluctuating_normal": ("最终正常", "波动后正常 (有反复，最终正常)"),
        "C1_late_hypo":          ("最终甲减", "迟发甲减 (曾正常，后转甲减)"),
        "C2_direct_hypo":        ("最终甲减", "直接甲减 (从未正常过，直接甲减)"),
        "C3_fluctuating_hypo":   ("最终甲减", "波动后甲减"),
    }

    print(f"\n{'='*70}")
    print(f"  轨迹分类 (互斥，合计={n})")
    print(f"{'='*70}")

    current_group = None
    group_total = 0
    for key in ["A1_persistent_hyper", "A2_relapsed_hyper",
                 "B1_stable_normal", "B2_improving_normal", "B3_fluctuating_normal",
                 "C1_late_hypo", "C2_direct_hypo", "C3_fluctuating_hypo"]:
        group, label = LABEL_MAP[key]
        cnt = (cat_series == key).sum()

        if group != current_group:
            if current_group is not None:
                print(f"  {'':>34} 小计: {group_total}")
                print()
            current_group = group
            group_total = 0
            print(f"  ── {group} ──")

        group_total += cnt
        bar = "█" * int(cnt / n * 40)
        print(f"    {label:<32s} {cnt:>4} ({cnt/n:>5.1%})  {bar}")

    print(f"  {'':>34} 小计: {group_total}")
    print(f"\n  {'─'*60}")

    total = sum((cat_series == k).sum() for k in LABEL_MAP)
    unknown = n - total
    if unknown > 0:
        print(f"    {'未分类':<32s} {unknown:>4}")
    print(f"    {'合计':<32s} {n:>4} (100%)")

    # ========== 复发详情 ==========
    relapse_idx = cat_series[cat_series == "A2_relapsed_hyper"].index
    print(f"\n{'='*70}")
    print(f"  复发甲亢详情 (n={len(relapse_idx)})")
    print(f"{'='*70}")

    # 首次好转时间
    first_ctrl = []
    for i in relapse_idx:
        seq = get_trajectory(eval_seq.iloc[i])
        for t, s in seq:
            if s in [2, 3]:
                first_ctrl.append(t)
                break

    print(f"  首次好转时间分布:")
    for t, cnt in sorted(Counter(first_ctrl).items(), key=lambda x: TIME_LABELS.index(x[0])):
        print(f"    {t}: {cnt}例")

    # ========== 多次治疗 ==========
    print(f"\n{'='*70}")
    print(f"  多次治疗患者 (n={len(retreatment)})")
    print(f"{'='*70}")

    transitions = Counter()
    for p in retreatment:
        first = p['outcomes'][0]
        last = p['outcomes'][-1]
        transitions[(first, last)] += 1

    print(f"  首次→末次结局:")
    for (f, t), cnt in sorted(transitions.items(), key=lambda x: -x[1]):
        print(f"    {STATUS.get(f,'?')} → {STATUS.get(t,'?')}: {cnt}例")

    rounds_dist = Counter(len(p['outcomes']) for p in retreatment)
    print(f"\n  治疗次数: ", end="")
    print(", ".join(f"{r}次={c}人" for r, c in sorted(rounds_dist.items())))

    # 详细列表（调用 print_retreatment_details 查看）


def print_retreatment_details(retreatment):
    """打印多次治疗患者的逐条详情（默认不执行，需要时手动调用）"""
    print(f"\n  多次治疗详细列表:")
    print(f"  {'ID':>8} {'年龄':>4} {'次数':>4}  {'结局变化':<20s}  {'剂量变化'}")
    for p in retreatment:
        o_str = " → ".join(STATUS.get(o, '?') for o in p['outcomes'])
        d_str = " → ".join(str(d) for d in p['doses'])
        print(f"  {str(p['id']):>8} {str(p['age']):>4} {len(p['outcomes']):>4}  {o_str:<20s}  {d_str}")


def analyze_transitions(df):
    """各相邻时间点之间的状态跳变统计"""
    eval_seq = df.iloc[:, EVAL_COLS].apply(pd.to_numeric, errors='coerce')
    eval_seq.columns = TIME_LABELS

    pairs = list(zip(TIME_LABELS[:-1], TIME_LABELS[1:]))
    # e.g. [("1M","3M"), ("3M","6M"), ("6M","1Y"), ("1Y","1.5Y")]

    print(f"\n{'='*70}")
    print(f"  各时间段状态跳变统计")
    print(f"{'='*70}")

    for t_from, t_to in pairs:
        col_from = eval_seq[t_from]
        col_to = eval_seq[t_to]
        both_valid = col_from.notna() & col_to.notna()
        n_valid = both_valid.sum()

        trans = Counter()
        for i in both_valid[both_valid].index:
            s_from = int(col_from.iloc[i])
            s_to = int(col_to.iloc[i])
            trans[(s_from, s_to)] += 1

        # 不变 vs 跳变
        unchanged = sum(v for (f, t), v in trans.items() if f == t)
        changed = sum(v for (f, t), v in trans.items() if f != t)

        print(f"\n  {t_from} → {t_to}  (有效配对: {n_valid})")
        print(f"  不变: {unchanged} ({unchanged/n_valid:.1%})  跳变: {changed} ({changed/n_valid:.1%})")
        print(f"  {'跳变类型':<16s} {'人数':>5} {'占比':>7}  {'方向'}")

        for (f, t), cnt in sorted(trans.items(), key=lambda x: -x[1]):
            if f == t:
                continue
            pct = cnt / n_valid
            if f == 1 and t in [2, 3]:
                direction = "↗ 好转"
            elif f in [2, 3] and t == 1:
                direction = "↘ 复发/恶化"
            elif f == 3 and t == 2:
                direction = "↘ 迟发甲减"
            elif f == 2 and t == 3:
                direction = "↗ 恢复正常"
            else:
                direction = "→ 变化"
            bar = "█" * int(pct * 40)
            print(f"    {STATUS[f]}→{STATUS[t]:<8s} {cnt:>5} ({pct:>5.1%})  {direction}  {bar}")

    # 汇总跳变方向
    print(f"\n{'='*70}")
    print(f"  跳变方向汇总 (所有时间段合计)")
    print(f"{'='*70}")

    all_trans = Counter()
    for t_from, t_to in pairs:
        col_from = eval_seq[t_from]
        col_to = eval_seq[t_to]
        both_valid = col_from.notna() & col_to.notna()
        for i in both_valid[both_valid].index:
            s_from = int(col_from.iloc[i])
            s_to = int(col_to.iloc[i])
            if s_from != s_to:
                all_trans[(s_from, s_to)] += 1

    total_changes = sum(all_trans.values())
    print(f"  总跳变次数: {total_changes}")
    for (f, t), cnt in sorted(all_trans.items(), key=lambda x: -x[1]):
        print(f"    {STATUS[f]} → {STATUS[t]}: {cnt} ({cnt/total_changes:.1%})")


if __name__ == "__main__":
    df, _ = load_all()
    main()
    analyze_transitions(df)
