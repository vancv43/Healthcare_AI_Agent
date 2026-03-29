import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# Giả lập config nếu bạn chưa có file config.py
PATIENT_MEMORY_PATH = "storage/patient_memory.json"
STORAGE_DIR = "storage"

def generate_patient_id(name: str, email: str) -> str:
    key = f"{name.strip().lower()}|{email.strip().lower()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def load_memory() -> Dict[str, Any]:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    if not os.path.exists(PATIENT_MEMORY_PATH):
        return {}
    try:
        with open(PATIENT_MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_memory(data: Dict[str, Any]) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(PATIENT_MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_patient_history(patient_id: str) -> Optional[Dict[str, Any]]:
    memory = load_memory()
    return memory.get(patient_id)

def _parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def _is_same_recent_visit(last_entry: Dict[str, Any], new_symptoms: str, window_minutes: int = 5) -> bool:
    last_symptoms = str(last_entry.get("symptoms", "")).strip().lower()
    current_symptoms = str(new_symptoms).strip().lower()
    if last_symptoms != current_symptoms:
        return False
        
    last_timestamp = _parse_timestamp(last_entry.get("timestamp", ""))
    if last_timestamp is None:
        return False
        
    now = datetime.now(timezone.utc)
    delta_minutes = (now - last_timestamp).total_seconds() / 60.0
    return delta_minutes <= window_minutes

def update_patient_memory(
    patient_id: str,
    demographics: Dict[str, Any],
    symptoms: str,
    diagnosis_result: Dict[str, Any],
    insurance_price: Optional[float],
    doctors: List[Dict[str, Any]],
) -> None:
    memory = load_memory()
    if patient_id not in memory:
        memory[patient_id] = {
            "demographics": demographics,
            "history": []
        }
        
    # always refresh demographics to latest values
    memory[patient_id]["demographics"] = demographics
    
    new_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symptoms": symptoms,
        "diagnosis": diagnosis_result,
        "insurance_price": insurance_price,
        "doctors": doctors,
    }
    
    history = memory[patient_id]["history"]
    
    if history:
        last_entry = history[-1]
        if _is_same_recent_visit(last_entry, symptoms, window_minutes=5):
            history[-1] = new_entry
            save_memory(memory)
            return
            
    history.append(new_entry)
    save_memory(memory)