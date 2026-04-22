
import re
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "NghiencuuHFLC.csv"
MODEL_DIR = BASE / "models"

MEASURE_ALIASES = {
    "bạch cầu": "bachcau", "bach cau": "bachcau", "bachcau": "bachcau",
    "tiểu cầu": "tieucau", "tieu cau": "tieucau", "tieucau": "tieucau",
    "hflc": "HFLC", "%hflc": "phantramHFLC", "phần trăm hflc": "phantramHFLC", "phan tram hflc": "phantramHFLC",
    "hct": "Hct", "hematocrit": "Hct",
    "độ tuổi": "Tuoi", "do tuoi": "Tuoi", "tuổi": "Tuoi", "tuoi": "Tuoi",
}

SYMPTOM_ALIASES = {
    "nôn ói": "NVnonoi", "non oi": "NVnonoi", "nonoi": "NVnonoi",
    "nonramau": "non ra mau",
    "tiêu chảy": "NVtieuchay", "tieu chay": "NVtieuchay", "tieuchay": "NVtieuchay",
    "đau bụng": "NVdaubung", "dau bung": "NVdaubung", "daubung": "NVdaubung",
    "đau cơ": "NVdauco", "dau co": "NVdauco", "dauco": "NVdauco",
    "đau đầu": "NVdaudau", "dau dau": "NVdaudau", "daudau": "NVdaudau",
    "đau sau hốc mắt": "NVdausauhocmat", "dau sau hoc mat": "NVdausauhocmat", "dausauhocmat": "NVdausauhocmat",
    "ho": "NVho",
    "xuất huyết": "NVxuathuyet", "xuat huyet": "NVxuathuyet",
    "gan to": "NVganto", "gan lớn": "NVganto",
    "petechia": "Petechia", "ban xuat huyet": "Petechia",
}

CANONICAL_MAP = {
    "petechiae": "petechia", "petechia": "petechia",
    "amdao": "am dao", "xhamdao": "am dao", "am dao": "am dao",
    "chaymaumui": "chay mau mui", "chay mau mui": "chay mau mui"," mui": "chay mau mui",
    "bammauvetchich": "vet chich", "vetchich": "vet chich", "vet chich": "vet chich",
    "chan rang": "chay mau rang", "chanrang": "chay mau rang", "chaymaurang": "chay mau rang", "chay mau rang": "chay mau rang",
    "tha": "tang huyet ap", "tăng huyết áp": "tang huyet ap", "tang huyet ap": "tang huyet ap", "tanghuyetap": "tang huyet ap",
    "dtd": "dai thao duong", "đái tháo đường": "dai thao duong", "dai thao duong": "dai thao duong", "daithaoduong": "dai thao duong",
    "roiloantiendinh": "roi loan tien dinh",
    "dongkinh" : "dong kinh",
    "tramcam": "tram cam",
    "thieumautanhuyet": "thieu mau tan huyet",
}

VIETNAMESE_TEXT_LABELS = {
    "petechia": "ban xuất huyết dạng chấm (petechia)",
    "am dao": "âm đạo",
    "chay mau mui": "chảy máu mũi",
    "vet chich": "vết chích / bầm máu tại vết chích",
    "chay mau rang": "chảy máu chân răng",
    "tang huyet ap": "tăng huyết áp",
    "dai thao duong": "đái tháo đường",
    "roi loan tien dinh": "rối loạn tiền đình",
    "dong kinh": "động kinh",
    "tram cam": "trầm cảm",
    "thieu mau tan huyet": "thiếu máu tan huyết",

}

SEVERITY_ALIASES = {
    "sốc": "sốc / nặng", "soc": "sốc / nặng", "nặng": "sốc / nặng", 
    "cảnh báo": "cảnh báo sốt xuất huyết", "canh bao": "cảnh báo sốt xuất huyết",
    "sxh": "sốt xuất huyết",
}

def simplify_severity(x):
    x = str(x).strip().lower()
    if x in {"", "nan"}:
        return "không rõ"
    if "soc" in x or "sox" in x:
        return "sốc / nặng"
    if "canhbao" in x:
        return "cảnh báo sốt xuất huyết"
    if x == "sxh":
        return "sốt xuất huyết"
    return "khác"

def map_gender(x):
    s = str(x).strip()
    if s == "1":
        return "nam"
    if s == "2":
        return "nữ"
    return s if s not in {"", "nan"} else "không rõ"

def normalize_daily_measure(base_name: str) -> str:
    key = base_name.strip().lower()
    mapping = {
        "bachcau": "bachcau", "tieucau": "tieucau", "hflc": "HFLC", "phantramhflc": "phantramHFLC", "hct": "Hct",
    }
    return mapping.get(key, base_name)
def format_filters_vi(filters: dict) -> str:
        parts = []

        if "severity_group" in filters:
            parts.append(f"nhóm {filters['severity_group']}")

        if "gioitinh_label" in filters:
            parts.append(f"giới tính {filters['gioitinh_label']}")

        if "tiencansxh" in filters:
            if filters["tiencansxh"] == 1:
                parts.append("có tiền căn SXH")
            elif filters["tiencansxh"] == 0:
                parts.append("không có tiền căn SXH")

        if not parts:
            return "toàn bộ bệnh nhân"

        return ", ".join(parts)
def prettify_measure(m):
    return {"bachcau":"bạch cầu", "tieucau":"tiểu cầu", "HFLC":"HFLC", "phantramHFLC":"%HFLC", "Hct":"Hct", "Tuoi":"tuổi"}.get(m, m)

def prettify_symptom(c):
    return {
        "NVnonoi":"nôn ói", "NVtieuchay":"tiêu chảy", "NVdaubung":"đau bụng", "NVdauco":"đau cơ",
        "NVdaudau":"đau đầu", "NVdausauhocmat":"đau sau hốc mắt", "NVho":"ho",
        "NVxuathuyet":"xuất huyết", "NVganto":"gan to", "Petechia":"ban xuất huyết dạng chấm"
    }.get(c, c)

def norm_basic_text(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace(";", ",")
    x = re.sub(r"\s+", " ", x)
    return x

def clean_token(token: str) -> str:
    t = norm_basic_text(token)
    t = t.replace(".", "")
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return CANONICAL_MAP.get(t, t)

def display_token(token: str) -> str:
    return VIETNAMESE_TEXT_LABELS.get(token, token)

def split_multi_value(x):
    if pd.isna(x):
        return []
    x = str(x).strip().lower()
    if x in {"", "nan"}:
        return []
    x = x.replace(";", ",")
    return [p.strip() for p in x.split(",") if p.strip()]

def clean_multivalue_cell(x):
    if pd.isna(x):
        return x
    s = norm_basic_text(x)
    if s in {"", "nan"}:
        return x
    tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
    cleaned = [clean_token(tok) for tok in tokens]
    seen, deduped = set(), []
    for tok in cleaned:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)
    return ", ".join(deduped)

def trust_level(rate):
    if rate >= 0.70:
        return "cao"
    elif rate >= 0.40:
        return "trung bình"
    elif rate >= 0.20:
        return "thấp"
    return "rất thấp"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["patient_id"] = [f"BN_{i:03d}" for i in range(1, len(df) + 1)]
    df["severity_group"] = df["ghichutinhtrangbenhnang"].apply(simplify_severity)
    df["gioitinh_label"] = df["gioitinh"].apply(map_gender) if "gioitinh" in df.columns else "không rõ"
    return df

def build_cleaned_df(df):
    df2 = df.copy()
    for col in ["Vitrixuathuyet", "Trieuchungkhac", "benhlynen"]:
        if col in df2.columns:
            df2[col] = df2[col].apply(clean_multivalue_cell)
    return df2

def build_before_after_table(df_raw, df_clean):
    rows = []
    for col in ["Vitrixuathuyet", "Trieuchungkhac", "benhlynen"]:
        if col not in df_raw.columns or col not in df_clean.columns:
            continue
        raw_tokens, clean_tokens = [], []
        for x in df_raw[col].dropna():
            raw_tokens.extend(split_multi_value(x))
        for x in df_clean[col].dropna():
            clean_tokens.extend(split_multi_value(x))
        raw_vc = pd.Series(raw_tokens).value_counts() if raw_tokens else pd.Series(dtype=int)
        clean_vc = pd.Series(clean_tokens).value_counts() if clean_tokens else pd.Series(dtype=int)
        all_idx = sorted(set(raw_vc.index).union(set(clean_vc.index)))
        for token in all_idx:
            rows.append([col, token, display_token(token), int(raw_vc.get(token, 0)), int(clean_vc.get(token, 0))])
    return pd.DataFrame(rows, columns=["column_name", "token", "nhan_tieng_viet", "count_truoc_lam_sach", "count_sau_lam_sach"])

def build_daily_tables(df):
    pattern = re.compile(r"^(.*?)[Nn](\d+)$")
    rows = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            measure = normalize_daily_measure(m.group(1))
            day = int(m.group(2))
            ser = pd.to_numeric(df[c], errors="coerce").dropna()
            rows.append([measure, day, c, len(ser), len(df), len(ser)/len(df),
                         ser.quantile(0.25) if len(ser) else None,
                         ser.quantile(0.50) if len(ser) else None,
                         ser.quantile(0.75) if len(ser) else None,
                         ser.median() if len(ser) else None,
                         ser.mean() if len(ser) else None,
                         ser.std() if len(ser) > 1 else None,
                         ser.min() if len(ser) else None,
                         ser.max() if len(ser) else None])
    daily_stats = pd.DataFrame(rows, columns=[
        "measure", "hospital_day", "original_column", "non_null_count", "total_count", "coverage_rate",
        "Q1", "Q2", "Q3", "median", "average", "std", "min", "max"
    ]).sort_values(["measure", "hospital_day"])
    daily_stats["trust_level"] = daily_stats["coverage_rate"].apply(trust_level)
    daily_stats["measure_label"] = daily_stats["measure"].apply(prettify_measure)
    return daily_stats

def build_symptom_tables(df):
    subjective = ["NVnonoi", "NVtieuchay", "NVdaubung", "NVdauco", "NVdaudau", "NVdausauhocmat", "NVho"]
    objective = ["NVxuathuyet", "NVganto", "Petechia"]
    def table(cols, group_name):
        rows = []
        for c in cols:
            present = int(df[c].notna().sum()) if c in df.columns else 0
            rows.append([group_name, c, prettify_symptom(c), present, len(df), present/len(df)])
        return pd.DataFrame(rows, columns=["group", "variable", "nhan_tieng_viet", "present_count", "total_patients", "lower_bound_rate"])
    return table(subjective, "triệu chứng nhập viện"), table(objective, "dấu hiệu nhập viện")

def build_text_token_tables(df):
    rows = []
    for c in ["Vitrixuathuyet", "benhlynen", "Trieuchungkhac"]:
        if c not in df.columns:
            continue
        tokens = []
        for val in df[c]:
            tokens.extend(split_multi_value(val))
        vc = pd.Series(tokens).value_counts()
        for token, count in vc.items():
            rows.append([c, token, display_token(token), int(count), count/len(df)])
    return pd.DataFrame(rows, columns=["column_name", "token", "nhan_tieng_viet", "count", "rate"])

def build_age_stats(df):
    ser = pd.to_numeric(df["Tuoi"], errors="coerce").dropna()
    return {
        "non_null_count": int(ser.shape[0]), "mean": float(ser.mean()) if len(ser) else None,
        "median": float(ser.median()) if len(ser) else None, "q1": float(ser.quantile(0.25)) if len(ser) else None,
        "q2": float(ser.quantile(0.50)) if len(ser) else None, "q3": float(ser.quantile(0.75)) if len(ser) else None,
        "min": float(ser.min()) if len(ser) else None, "max": float(ser.max()) if len(ser) else None,
        "std": float(ser.std()) if len(ser) > 1 else None, "coverage_rate": float(ser.shape[0] / len(df)) if len(df) else None,
    }

def build_history_tables(df):
    rows = []
    for c in ["tiencansxh", "tiencanbenhlykhac", "pregnancy"]:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False)
            for val, cnt in vc.items():
                rows.append([c, str(val), int(cnt), cnt/len(df)])
    return pd.DataFrame(rows, columns=["column_name", "value", "count", "rate"])

def parse_filters(question: str):
    q = question.lower()
    filters = {}

    if "nữ" in q or " nu " in f" {q} " or q.endswith(" nu") or "giới tính nữ" in q:
        filters["gioitinh_label"] = "nữ"
    elif "nam" in q or "giới tính nam" in q:
        filters["gioitinh_label"] = "nam"

    q_for_severity = q
    if "tiền căn sxh" in q or "tien can sxh" in q:
        if "không" in q or "khong" in q:
            filters["tiencansxh"] = 0
        else:
            filters["tiencansxh"] = 1
        q_for_severity = q_for_severity.replace("tiền căn sxh", " ").replace("tien can sxh", " ")


    for alias, sev in SEVERITY_ALIASES.items():
        if alias in q_for_severity:
            filters["severity_group"] = sev
            break

    return filters
def apply_filters(df, filters: dict):
    out = df.copy()
    for k, v in filters.items():
        if k not in out.columns:
            continue
        if k == "tiencansxh":
            out = out[pd.to_numeric(out[k], errors="coerce") == v]
        else:
            out = out[out[k].astype(str).str.lower() == str(v).lower()]
    return out

def extract_day(question: str):
    q = question.lower()
    m = re.search(r"ngày\s*(\d+)", q)
    if m: return int(m.group(1))
    m = re.search(r"\bn\s*(\d+)\b", q)
    if m: return int(m.group(1))
    return None

def extract_measure(question: str):
    q = question.lower()
    for alias, canonical in sorted(MEASURE_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            return canonical
    return None

def extract_symptom(question: str):
    q = question.lower()
    for alias, canonical in sorted(SYMPTOM_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            return canonical
    return None
def rule_based_intent(question: str):
    q = question.lower().strip()

    bleeding_site_patterns = [
        "vị trí xuất huyết",
        "vi tri xuat huyet",
        "xuất huyết ở vị trí nào",
        "xuat huyet o vi tri nao",
        "chảy máu ở vị trí nào",
        "chay mau o vi tri nao",
    ]
    for p in bleeding_site_patterns:
        if p in q:
            return "top_bleeding_site"

    symptom_markers = [
        "nôn ói", "non oi", "nonoi",
        "tiêu chảy", "tieu chay", "tieuchay",
        "đau bụng", "dau bung", "daubung",
        "đau cơ", "dau co", "dauco",
        "đau đầu", "dau dau", "daudau",
        "đau sau hốc mắt", "dau sau hoc mat", "dausauhocmat",
        "ho",
        "triệu chứng", "trieu chung",
        "xuất huyết", "xuat huyet",
    ]
    has_symptom_marker = any(m in q for m in symptom_markers)
    has_cohort_marker = (
        "ở bệnh nhân" in q or "o benh nhan" in q or
        "ở nhóm" in q or "o nhom" in q or
        "giới tính" in q or "gioi tinh" in q or
        "nữ" in q or " nam " in f" {q} " or
        "tiền căn sxh" in q or "tien can sxh" in q
    )


    if has_cohort_marker and has_symptom_marker:
        return "cohort_query"

    if "tiền căn sxh" in q or "tien can sxh" in q:
        return "history_stats"

    if "giới tính" in q or "gioi tinh" in q or "nam nữ" in q or "nam nu" in q:
        return "gender_stats"

    if "độ tuổi" in q or "do tuoi" in q or "tuổi trung bình" in q or "tuoi trung binh" in q:
        return "age_stats"

    if "bệnh lý nền" in q or "benh ly nen" in q or "comorbidity" in q:
        return "top_comorbidity"

    if "so sánh" in q or "so sanh" in q:
        return "compare_query"

    overview_patterns = [
        "có bao nhiêu bệnh nhân",
        "co bao nhieu benh nhan",
        "tổng số bệnh nhân",
        "tong so benh nhan",
        "bao nhiêu ca",
        "bao nhieu ca",
    ]
    if not has_symptom_marker and not has_cohort_marker:
        for p in overview_patterns:
            if p in q:
                return "overview_count"

    return None