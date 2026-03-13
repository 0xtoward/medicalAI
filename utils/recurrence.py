"""Shared recurrence-oriented data builders."""

import numpy as np
import pandas as pd

from utils.config import STATIC_NAMES, STATE_NAMES, TIME_STAMPS

TIME_MONTHS = [0, 1, 3, 6, 12, 18, 24]


def build_interval_risk_data(X_s, X_d, S_matrix, pids, target_k=None, row_ids=None):
    """Build interval-level at-risk rows for current Normal -> next-window Hyper."""
    rows = []
    n_samples, n_timepoints = S_matrix.shape
    k_range = [target_k] if target_k is not None else range(n_timepoints - 1)
    if row_ids is None:
        row_ids = np.arange(n_samples)

    for i in range(n_samples):
        pid = pids[i]
        row_id = int(row_ids[i])
        xs = X_s[i]
        for k in k_range:
            if int(S_matrix[i, k]) != 1:
                continue

            labs_k = X_d[i, k, :]
            labs_0 = X_d[i, 0, :]
            y_relapse = int(S_matrix[i, k + 1] == 0)
            next_state = int(S_matrix[i, k + 1])
            hist_states = S_matrix[i, :k]
            prev_state = int(S_matrix[i, k - 1]) if k > 0 else -1
            prior_relapse_count = int(
                sum(int(S_matrix[i, m] == 1 and S_matrix[i, m + 1] == 0) for m in range(k))
            )
            ever_hyper = int(0 in hist_states)
            ever_hypo = int(2 in hist_states)
            time_in_normal = int(np.sum(hist_states == 1))

            delta_ft4_k0 = labs_k[1] - labs_0[1]
            delta_tsh_k0 = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(
                np.clip(labs_0[2], 0, None)
            )
            if k > 0:
                labs_prev = X_d[i, k - 1, :]
                delta_ft4_1step = labs_k[1] - labs_prev[1]
                delta_tsh_1step = np.log1p(np.clip(labs_k[2], 0, None)) - np.log1p(
                    np.clip(labs_prev[2], 0, None)
                )
            else:
                delta_ft4_1step = 0.0
                delta_tsh_1step = 0.0

            rows.append(
                {
                    "Patient_ID": pid,
                    "Source_Row": row_id,
                    "Interval_ID": k,
                    "Interval_Name": f"{TIME_STAMPS[k]}->{TIME_STAMPS[k + 1]}",
                    "Start_Time": TIME_MONTHS[k],
                    "Stop_Time": TIME_MONTHS[k + 1],
                    "Interval_Width": TIME_MONTHS[k + 1] - TIME_MONTHS[k],
                    "Y_Relapse": y_relapse,
                    "Next_State": STATE_NAMES[next_state],
                    "Prior_Relapse_Count": prior_relapse_count,
                    "Event_Order": prior_relapse_count + 1,
                    **dict(zip(STATIC_NAMES, xs)),
                    "FT3_Current": labs_k[0],
                    "FT4_Current": labs_k[1],
                    "logTSH_Current": np.log1p(np.clip(labs_k[2], 0, None)),
                    "Prev_State": str(prev_state),
                    "Ever_Hyper_Before": ever_hyper,
                    "Ever_Hypo_Before": ever_hypo,
                    "Time_In_Normal": time_in_normal,
                    "Delta_FT4_k0": delta_ft4_k0,
                    "Delta_TSH_k0": delta_tsh_k0,
                    "Delta_FT4_1step": delta_ft4_1step,
                    "Delta_TSH_1step": delta_tsh_1step,
                }
            )
    cols = [
        "Patient_ID",
        "Source_Row",
        "Interval_ID",
        "Interval_Name",
        "Start_Time",
        "Stop_Time",
        "Interval_Width",
        "Y_Relapse",
        "Next_State",
        "Prior_Relapse_Count",
        "Event_Order",
        *STATIC_NAMES,
        "FT3_Current",
        "FT4_Current",
        "logTSH_Current",
        "Prev_State",
        "Ever_Hyper_Before",
        "Ever_Hypo_Before",
        "Time_In_Normal",
        "Delta_FT4_k0",
        "Delta_TSH_k0",
        "Delta_FT4_1step",
        "Delta_TSH_1step",
    ]
    return pd.DataFrame(rows, columns=cols)


def derive_recurrent_survival_data(interval_df):
    """Collapse contiguous Normal intervals into recurrent risk spells."""
    if len(interval_df) == 0:
        return pd.DataFrame()

    rows = []
    for pid, g in interval_df.groupby("Patient_ID", sort=False):
        g = g.sort_values("Interval_ID").reset_index(drop=True)
        spell_id = 0
        spell_starts = []
        for idx in range(len(g)):
            if idx == 0 or int(g.loc[idx, "Interval_ID"]) != int(g.loc[idx - 1, "Interval_ID"]) + 1:
                spell_id += 1
            spell_starts.append(spell_id)
        g = g.copy()
        g["Spell_ID"] = spell_starts

        for _, spell in g.groupby("Spell_ID", sort=False):
            first = spell.iloc[0]
            last = spell.iloc[-1]
            event = int(spell["Y_Relapse"].max() > 0)
            if event:
                end_type = "Relapse"
            elif last["Next_State"] == "Hypo":
                end_type = "Hypo_Censor"
            else:
                end_type = "Administrative_Censor"

            row = first.to_dict()
            row["Spell_Start_Time"] = float(first["Start_Time"])
            row["Spell_Stop_Time"] = float(last["Stop_Time"])
            row["Gap_Time"] = float(last["Stop_Time"] - first["Start_Time"])
            row["Event"] = event
            row["End_Type"] = end_type
            row["Is_Spell_Start"] = 1
            row["Y_Relapse_Next"] = int(first["Y_Relapse"])
            row["Next_Window_Months"] = float(first["Interval_Width"])
            row["Spell_Intervals"] = int(len(spell))
            rows.append(row)

    return pd.DataFrame(rows)
