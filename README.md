
# HFLC QA Model Project - UI v5

Bản v5 bổ sung **GraphRAG-lite**:
- query router
- cohort filters từ câu hỏi
- hỏi cohort phức hợp như:
  - "ở bệnh nhân nữ, nonoi có bao nhiêu"
  - "ở bệnh nhân có tiền căn sxh, nonoi có bao nhiêu"
  - "ở nhóm sốc, triệu chứng nào hay gặp"

## Chạy
```bash
pip install -r requirements.txt
python src/train_intent_model.py
streamlit run app.py
```
