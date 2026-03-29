# Healthcare_AI_Agent

healthcare_agent/
├── core/                  # Chứa logic xử lý cốt lõi
│   ├── explainer.py       # Lớp AI dịch thuật kết quả bằng FLAN-T5
│   └── memory.py          # Xử lý lưu trữ lịch sử bệnh nhân
├── ui/                    # Chứa giao diện người dùng
│   ├── __init__.py
│   └── app.py             # File khởi chạy Gradio UI
├── storage/               # Lưu trữ dữ liệu hệ thống sinh ra (patient_memory.json)
└── main.py                # Cấu hình AI Database, Doctor, Insurance models

# 🏥 Healthcare AI Agent

[cite_start]Dự án **Healthcare AI Agent** là một hệ thống trí tuệ nhân tạo hỗ trợ y tế đa lớp[cite: 21, 26]. Hệ thống bao gồm:
1. [cite_start]**Tìm kiếm ngữ nghĩa (Semantic Search):** Sử dụng FAISS và SentenceTransformers để tìm kiếm các mẫu bệnh lý[cite: 175, 237, 238].
2. [cite_start]**Hệ thống chuyên gia:** Gợi ý bác sĩ và ước tính chi phí bảo hiểm bằng Machine Learning (Random Forest)[cite: 216, 219, 241, 242, 249, 250].
3. [cite_start]**Lớp giải thích AI (AI Explainer):** Sử dụng LLM (`google/flan-t5-large`) để diễn giải kết quả y khoa thành ngôn ngữ dễ hiểu, thân thiện với bệnh nhân[cite: 253, 254].
4. [cite_start]**Giao diện người dùng:** Xây dựng bằng Gradio[cite: 257, 258].

---

## 🚀 Hướng dẫn Cài đặt và Khởi chạy

### 1. Yêu cầu hệ thống
- [cite_start]Python 3.9 trở lên (Khuyến nghị dùng bản 3.10 đến 3.13)[cite: 76].

### 2. Thiết lập môi trường (Setup Environment)
[cite_start]Mở Terminal/Command Prompt tại thư mục gốc của dự án (thư mục `healthcare_agent/`) [cite: 71, 72] và chạy các lệnh sau:

**Bước 2.1: Tạo môi trường ảo (Virtual Environment)**
```bash
python -m venv venv
# Healthcare_AI_Agent
