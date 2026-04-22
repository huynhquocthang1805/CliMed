
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qa_engine import HFLCQAEngine
from _common import apply_filters,  parse_filters, format_filters_vi

st.set_page_config(page_title="HFLC AI Assistant v5", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_engine():
    return HFLCQAEngine()

engine = load_engine()

def get_rec(trust_level: str) -> str:
    return {
        "cao": "Có thể dùng cho dự đoán.",
        "trung bình": "Tham khảo thêm coverage",
        "thấp": "Chỉ nên tham khảo.",
        "rất thấp": "Chưa thể đưa ra kết luận"
    }.get(trust_level, "")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1320px;}
.ai-card {padding: 14px 16px; border-radius: 16px; background: #f6f8fb; border: 1px solid rgba(49,51,63,0.08);}
.small-muted {color: #6b7280; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🩺 AI Assistant")

    st.divider()
    st.code("python src/train_intent_model.py", language="bash")
    st.code("streamlit run app.py", language="bash")
    st.subheader("Ví dụ nhanh")
    examples = [
        "ở bệnh nhân nữ, có bao nhiêu bệnh nhân nôn ói",
        "ở bệnh nhân có tiền căn sxh, triệu chứng nào hay gặp",
        "ở bệnh nhân nữ có tiền căn sxh, triệu chứng nào hay gặp",
        "ở nhóm sốc, triệu chứng nào hay gặp",
        "so sánh nam và nữ",
        "vị trí xuất huyết nào gặp nhiều nhất",
        "HFLC ngày 5 trung vị bao nhiêu",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state["prefill_question"] = ex
    st.divider()


c1, c2, c3, c4 = st.columns(4)
c1.metric("Tổng bệnh nhân", len(engine.df))
c2.metric("Intent hỗ trợ", len(set(engine.intent_model.classes_)))
c3.metric("Số chỉ số theo ngày", engine.daily_stats["measure"].nunique())
c4.metric("Tuổi trung bình", f"{engine.age_stats['mean']:.1f}")

st.markdown('<div class="ai-card"><b>Trợ lý dữ liệu </b><br><span class="small-muted"></span></div>', unsafe_allow_html=True)
st.write("")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Xin chào, mình đã sẵn sàng trả lời câu hỏi của bạn"}]
if "prefill_question" not in st.session_state:
    st.session_state["prefill_question"] = ""

tab_chat, tab_router, tab_coverage, tab_data, tab_clean = st.tabs(["💬 Chat", "🧭 Router & Cohort", "📈 Coverage & Trust", "📊 Data Explorer", "🧹 Trước/Sau làm sạch"])

with tab_chat:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Nhập câu hỏi, ví dụ: ở bệnh nhân nữ có tiền căn sxh, triệu chứng nào hay gặp")
    injected = st.session_state.pop("prefill_question", "")
    if injected and not prompt:
        prompt = injected
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = engine.answer(prompt)
        debug_info = engine.debug_parse(prompt)
        intent = debug_info["intent"]
        filters = debug_info["filters"]
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Intent dự đoán: {intent} | Filters: {format_filters_vi(filters)}")
        st.session_state["messages"].append({"role": "assistant", "content": answer + f"\n\n*Intent dự đoán: {intent} | Filters: {filters}*"})

with tab_router:
    st.subheader("Giải thích query router và cohort filters")
    q = st.text_input("Thử nhập câu hỏi để xem router hiểu gì", value="ở bệnh nhân nữ có tiền căn sxh, triệu chứng nào hay gặp", key="router_demo")
    debug_info = engine.debug_parse(q)
    subset = apply_filters(engine.df, debug_info["filters"]) if debug_info["filters"] else engine.df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intent", debug_info["intent"])
    c2.metric("Số filter", len(debug_info["filters"]))
    c3.metric("Cỡ cohort", debug_info["cohort_size"])
    c4.metric("Route", debug_info["route"])
    st.markdown("**Thông tin parser**")
    st.json(debug_info)
    st.markdown("**Logic trả lời dự kiến**")
    logic_df = pd.DataFrame([{
        "question": debug_info["question"],
        "intent": debug_info["intent"],
        "filters": str(debug_info["filters"]),
        "symptom_extracted": debug_info["symptom_extracted"],
        "measure_extracted": debug_info["measure_extracted"],
        "day_extracted": debug_info["day_extracted"],
        "route": debug_info["route"],
        "cohort_size": debug_info["cohort_size"],
    }])
    st.dataframe(logic_df, use_container_width=True)
    if not subset.empty:
        st.markdown("**Preview cohort sau lọc**")
        show_cols = [c for c in ["patient_id", "Ten", "gioitinh_label", "severity_group", "tiencansxh", "Khoa"] if c in subset.columns]
        st.dataframe(subset[show_cols].head(30), use_container_width=True)

with tab_coverage:
    st.subheader("Coverage và độ tin cậy của các chỉ số theo ngày")
    measures = sorted(engine.daily_stats["measure"].dropna().unique().tolist())
    selected_measure = st.selectbox("Chọn chỉ số để xem chi tiết", measures, index=0)
    measure_df = engine.daily_stats[engine.daily_stats["measure"] == selected_measure].sort_values("hospital_day").copy()

    a, b = st.columns(2)
    with a:
        st.markdown("**Coverage theo ngày**")
        st.line_chart(measure_df.set_index("hospital_day")[["coverage_rate"]])
    with b:
        st.markdown("**Median và average theo ngày**")
        st.line_chart(measure_df.set_index("hospital_day")[["median", "average"]])

    display_df = measure_df[["hospital_day", "measure_label", "non_null_count", "total_count", "coverage_rate", "trust_level", "Q1", "Q2", "Q3", "median", "average", "std", "min", "max"]].copy()
    display_df["coverage_rate"] = (display_df["coverage_rate"] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True)

with tab_data:
    d0, d1, d2, d3, d4 = st.tabs(["Tổng quan", "Triệu chứng", "Dấu hiệu", "Độ tuổi & giới tính", "Tiền căn & text"])
    with d0:
        base_df = engine.df[["patient_id", "Ten", "Khoa", "Tuoi", "gioitinh_label", "severity_group"]].copy()
        base_df = base_df.rename(columns={"patient_id":"Mã BN", "Ten":"Tên", "Khoa":"Khoa", "Tuoi":"Tuổi", "gioitinh_label":"Giới tính", "severity_group":"Mức độ"})
        st.dataframe(base_df.head(148), use_container_width=True)
    with d1:
        temp = engine.symptoms_table.sort_values("present_count", ascending=False).copy()
        temp = temp.rename(columns={"nhan_tieng_viet":"Triệu chứng", "present_count":"Số ca ghi nhận", "total_patients":"Tổng bệnh nhân", "lower_bound_rate":"Tỷ lệ tối thiểu"})
        st.dataframe(temp[["Triệu chứng", "Số ca ghi nhận", "Tổng bệnh nhân", "Tỷ lệ tối thiểu"]], use_container_width=True)
        st.bar_chart(temp.set_index("Triệu chứng")[["Số ca ghi nhận"]])
    with d2:
        temp = engine.signs_table.sort_values("present_count", ascending=False).copy()
        temp = temp.rename(columns={"nhan_tieng_viet":"Dấu hiệu", "present_count":"Số ca ghi nhận", "total_patients":"Tổng bệnh nhân", "lower_bound_rate":"Tỷ lệ tối thiểu"})
        st.dataframe(temp[["Dấu hiệu", "Số ca ghi nhận", "Tổng bệnh nhân", "Tỷ lệ tối thiểu"]], use_container_width=True)
        st.bar_chart(temp.set_index("Dấu hiệu")[["Số ca ghi nhận"]])
    with d3:
        age = engine.age_stats
        age_df = pd.DataFrame([{"n": age["non_null_count"], "coverage": age["coverage_rate"], "mean": age["mean"], "median": age["median"], "Q1": age["q1"], "Q2": age["q2"], "Q3": age["q3"], "min": age["min"], "max": age["max"], "std": age["std"]}])
        st.markdown("**Thống kê độ tuổi**")
        st.dataframe(age_df, use_container_width=True)
        st.markdown("**Phân bố giới tính**")
        gender_df = engine.df["gioitinh_label"].value_counts(dropna=False).rename_axis("Giới tính").reset_index(name="Số ca")
        st.dataframe(gender_df, use_container_width=True)
        st.bar_chart(gender_df.set_index("Giới tính")[["Số ca"]])
    with d4:
        st.markdown("**Tiền căn SXH**")
        hist = engine.history_table[engine.history_table["column_name"] == "tiencansxh"].sort_values("count", ascending=False)
        hist = hist.rename(columns={"value":"Giá trị", "count":"Số ca", "rate":"Tỷ lệ"})
        st.dataframe(hist[["Giá trị", "Số ca", "Tỷ lệ"]], use_container_width=True)
        if not hist.empty:
            st.bar_chart(hist.set_index("Giá trị")[["Số ca"]])
        st.markdown("**Bệnh lý nền sau làm sạch**")
        temp = engine.text_table[engine.text_table["column_name"] == "benhlynen"].sort_values("count", ascending=False)
        st.dataframe(temp.rename(columns={"nhan_tieng_viet":"Nhãn tiếng Việt", "count":"Số ca", "rate":"Tỷ lệ"})[["Nhãn tiếng Việt", "Số ca", "Tỷ lệ"]], use_container_width=True)

with tab_clean:
    st.subheader("Bảng trước và sau khi làm sạch")
    choice = st.selectbox("Chọn cột text", ["Vitrixuathuyet", "Trieuchungkhac", "benhlynen"])
    temp = engine.before_after_table[engine.before_after_table["column_name"] == choice].sort_values(["count_sau_lam_sach", "count_truoc_lam_sach"], ascending=False)
    temp = temp.rename(columns={"token":"Token gốc / hoặc đã chuẩn hóa", "nhan_tieng_viet":"Nhãn tiếng Việt", "count_truoc_lam_sach":"Số ca trước làm sạch", "count_sau_lam_sach":"Số ca sau làm sạch"})
    st.dataframe(temp[["Token gốc / hoặc đã chuẩn hóa", "Nhãn tiếng Việt", "Số ca trước làm sạch", "Số ca sau làm sạch"]], use_container_width=True)


