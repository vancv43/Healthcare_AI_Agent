import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. KNOWLEDGE BASE & EMBEDDINGS (Semantic Search)
# Sử dụng SentenceTransformer và FAISS theo yêu cầu trong slide
# =====================================================================
class MedicalKnowledgeBase:
    def __init__(self):
        # Slide 10: Sử dụng model all-MiniLM-L6-v2 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Dữ liệu mô phỏng (Trong thực tế bạn sẽ load từ data/medical_knowledge.csv)
        self.knowledge_data = [
            {"symptoms": "fever + cough", "condition": "respiratory infection", "specialty": "internal medicine", "advice": "rest + hydration"},
            {"symptoms": "severe chest pain + shortness of breath", "condition": "heart issue", "specialty": "cardiology", "advice": "go to emergency immediately"},
            {"symptoms": "stomach ache + nausea after eating", "condition": "food poisoning", "specialty": "gastroenterology", "advice": "drink clear fluids + rest"}
        ]
        
        # Slide 11 & 12: Tạo FAISS Vector Database
        self.symptom_texts = [item["symptoms"] for item in self.knowledge_data]
        self.embeddings = self.model.encode(self.symptom_texts)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings))

    def search_symptoms(self, user_query):
        # Biến đổi câu hỏi thành vector và tìm kiếm
        query_vector = self.model.encode([user_query])
        distances, indices = self.index.search(np.array(query_vector), k=1)
        
        best_match_idx = indices[0][0]
        return self.knowledge_data[best_match_idx]

# =====================================================================
# 2. DOCTOR RECOMMENDATION
# Tìm kiếm bác sĩ dựa trên chuyên khoa và thành phố
# =====================================================================
class DoctorRecommender:
    def __init__(self):
        # Dataset mô phỏng (Homework: Thêm data vào database này)
        self.doctors = [
            {"name": "Dr. Smith", "specialty": "internal medicine", "city": "Bangkok", "rating": 4.8},
            {"name": "Dr. Davis", "specialty": "cardiology", "city": "Bangkok", "rating": 4.9},
            {"name": "Dr. Lee", "specialty": "internal medicine", "city": "Chiang Mai", "rating": 4.5}
        ]

    def find_doctor(self, specialty, city):
        matches = [doc for doc in self.doctors if doc["specialty"] == specialty and doc["city"] == city]
        return matches if matches else "No matching doctors found in your area."

# =====================================================================
# 3. INSURANCE PREDICTION (Machine Learning)
# =====================================================================
class InsurancePredictor:
    def __init__(self):
        # Slide 16: Dùng RandomForestRegressor với các inputs: Age, BMI, Smoking, Region, Children
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Tạo dữ liệu huấn luyện giả lập
        X_train = np.array([
            [25, 22.5, 0, 1, 0], # Trẻ, BMI thấp, không hút thuốc -> Rẻ
            [50, 30.0, 1, 2, 2], # Già, BMI cao, hút thuốc -> Đắt
            [35, 25.5, 0, 1, 1]  # Trung bình
        ])
        y_train = np.array([1500, 8500, 3200]) # Chi phí bảo hiểm (USD)
        
        self.model.fit(X_train, y_train)

    def estimate_cost(self, age, bmi, smoking, region_code, children):
        features = np.array([[age, bmi, smoking, region_code, children]])
        prediction = self.model.predict(features)
        return round(prediction[0], 2)

# =====================================================================
# 4. PATIENT MEMORY (Persistence)
# Lưu lịch sử khám bệnh bằng file JSON
# =====================================================================
class PatientMemory:
    def __init__(self, filepath="storage/patient_memory.json"):
        self.filepath = filepath
        # Đảm bảo thư mục storage tồn tại
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)

    def save_visit(self, patient_id, visit_data):
        with open(self.filepath, 'r') as f:
            memory = json.load(f)
            
        if patient_id not in memory:
            memory[patient_id] = []
        memory[patient_id].append(visit_data)
        
        with open(self.filepath, 'w') as f:
            json.dump(memory, f, indent=4)
            
    def get_history(self, patient_id):
        with open(self.filepath, 'r') as f:
            memory = json.load(f)
        return memory.get(patient_id, "No previous history found.")

# =====================================================================
# 5. THE AI AGENT PIPELINE (Kết hợp tất cả lại)
# =====================================================================
def run_healthcare_agent():
    print("Initializing Healthcare AI Agent (Session 1)...")
    knowledge_base = MedicalKnowledgeBase()
    doctor_db = DoctorRecommender()
    insurance_model = InsurancePredictor()
    memory = PatientMemory()

    print("\n--- NEW PATIENT VISIT ---")
    patient_id = "PATIENT_001"
    user_city = "Bangkok"
    user_symptoms = "I have a bad cough and feel feverish."
    
    # 1. Retrieval & Reasoning (Semantic Search)
    print(f"\n[1] Analyzing Symptoms: '{user_symptoms}'")
    insight = knowledge_base.search_symptoms(user_symptoms)
    print(f" -> Predicted Condition: {insight['condition']}")
    print(f" -> Advice: {insight['advice']}")
    
    # 2. Doctor Recommendation
    print(f"\n[2] Recommending Specialist ({insight['specialty']} in {user_city}):")
    doctors = doctor_db.find_doctor(insight['specialty'], user_city)
    for doc in doctors:
        print(f" -> {doc['name']} (Rating: {doc['rating']})")
        
    # 3. Insurance Prediction (Ví dụ: Tuổi 30, BMI 24.5, Không hút, Vùng 1, 0 con)
    print("\n[3] Estimating Insurance Cost:")
    est_cost = insurance_model.estimate_cost(age=30, bmi=24.5, smoking=0, region_code=1, children=0)
    print(f" -> Estimated Cost: ${est_cost}")
    
    # 4. Save to Memory
    print("\n[4] Saving visit to Patient Memory (JSON)...")
    visit_record = {
        "symptoms": user_symptoms,
        "condition": insight['condition'],
        "recommended_specialty": insight['specialty'],
        "insurance_estimate": est_cost
    }
    memory.save_visit(patient_id, visit_record)
    print(" -> Visit saved successfully!")

if __name__ == "__main__":
    run_healthcare_agent()  