import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------- Config ----------
st.set_page_config(
    page_title="College Student Placement Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_pipeline(pipeline_path: Path):
    return joblib.load(pipeline_path)


def get_feature_columns():
    # Pipeline için gerekli sütunlar
    return [
        "IQ",
        "Prev_Sem_Result",
        "CGPA",
        "Academic_Performance",
        "Internship_Experience",
        "Extra_Curricular_Score",
        "Communication_Skills",
        "Projects_Completed"
    ]


def make_input_dataframe(
    iq: int,
    prev_sem_result: float,
    cgpa: float,
    academic_performance: int,
    internship_experience: str,
    extra_curricular_score: int,
    communication_skills: int,
    projects_completed: int,
):
    # Pipeline için basit format
    data = {
        "IQ": int(iq),
        "Prev_Sem_Result": float(prev_sem_result),
        "CGPA": float(cgpa),
        "Academic_Performance": int(academic_performance),
        "Internship_Experience": internship_experience,
        "Extra_Curricular_Score": int(extra_curricular_score),
        "Communication_Skills": int(communication_skills),
        "Projects_Completed": int(projects_completed)
    }
    
    return pd.DataFrame([data])


def format_placement_result(value: int) -> str:
    if value == 1:
        return "✅ Yerleştirildi (Placed)"
    else:
        return "❌ Yerleştirilmedi (Not Placed)"


def main():
    st.title("🎓 College Student Placement Predictor")
    st.caption("Üniversite öğrencilerinin işe yerleştirilme durumunu tahmin etmek için basit ve modern bir arayüz.")

    pipeline_path = Path("model/pipe.pkl")
    if not pipeline_path.exists():
        st.error("Pipeline dosyası bulunamadı: pipe.pkl")
        st.stop()

    with st.spinner("Pipeline yükleniyor..."):
        pipeline = load_pipeline(pipeline_path)

    st.subheader("Öğrenci Bilgileri")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        
        with c1:
            iq = st.number_input("IQ Seviyesi", min_value=40, max_value=158, value=100, step=1)
            prev_sem_result = st.number_input("Önceki Dönem Sonucu", min_value=5.0, max_value=10.0, value=7.0, step=0.1)
            cgpa = st.number_input("CGPA (Genel Not Ortalaması)", min_value=5.0, max_value=10.0, value=7.0, step=0.1)
            academic_performance = st.number_input("Akademik Performans (1-10)", min_value=1, max_value=10, value=7, step=1)

        with c2:
            internship_experience = st.selectbox("Staj Deneyimi", options=["No", "Yes"], index=0)
            extra_curricular_score = st.number_input("Ders Dışı Etkinlik Skoru (1-10)", min_value=1, max_value=10, value=5, step=1)
            communication_skills = st.number_input("İletişim Becerileri (1-10)", min_value=1, max_value=10, value=7, step=1)
            projects_completed = st.number_input("Tamamlanan Proje Sayısı", min_value=0, max_value=10, value=2, step=1)

        submitted = st.form_submit_button("Yerleştirme Durumunu Tahmin Et")

    if submitted:
        try:
            X = make_input_dataframe(
                iq=int(iq),
                prev_sem_result=float(prev_sem_result),
                cgpa=float(cgpa),
                academic_performance=int(academic_performance),
                internship_experience=internship_experience,
                extra_curricular_score=int(extra_curricular_score),
                communication_skills=int(communication_skills),
                projects_completed=int(projects_completed),
            )

            # Pipeline ile tahmin yap
            pred = pipeline.predict(X)[0]

            st.success("Tahmin Başarılı")
            
            # Sonucu göster
            result_text = format_placement_result(pred)
            if pred == 1:
                st.success(f"🎉 {result_text}")
            else:
                st.warning(f"⚠️ {result_text}")

            # Olasılık skorunu da göster (eğer model predict_proba destekliyorsa)
            try:
                proba = pipeline.predict_proba(X)[0]
                st.info(f"Yerleştirilme Olasılığı: {proba[1]:.2%}")
            except:
                pass

            with st.expander("Girdi Özeti"):
                st.dataframe(X.T.rename(columns={0: "değer"}))

        except Exception as e:
            st.error("Tahmin sırasında bir hata oluştu. Ayrıntı için aşağıya bakın.")
            st.exception(e)


if __name__ == "__main__":
    main()


