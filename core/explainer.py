from typing import Dict, Any, Optional
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AIExplainer:
    """Bootcamp-friendly explanation layer."""
    def __init__(self, model_name: str = "google/flan-t5-large", use_local_model: bool = False):
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()
        
        if self.use_local_model:
            self._try_load_model()

    def _try_load_model(self):
        try:
            print(f"Loading explainer model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model_loaded = True
            print("Explainer model loaded successfully.")
        except Exception as e:
            self.model_loaded = False
            self.load_error = str(e)
            print("Explainer model could not be loaded. Falling back to deterministic explanation.")
            print(f"Reason: {self.load_error}")

    def enable_local_model(self):
        if not self.model_loaded:
            self.use_local_model = True
            self._try_load_model()

    def _safe_get_top(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload.get("top_diagnosis", {}) or {}

    def _build_structured_explanation(self, payload: Dict[str, Any]) -> str:
        top = self._safe_get_top(payload)
        doctors = payload.get("doctors", []) or []
        insurance_text = payload.get("insurance_text", "Not available.")
        history_summary = payload.get("history_summary", "")

        if not top:
            return (
                f"{history_summary}\n"
                "The system could not generate a clear health insight from the current input.\n"
                "Please review the symptoms entered and try again."
            )

        diagnosis = top.get("diagnosis", "Unknown")
        specialty = top.get("specialty", "Unknown")
        urgency = top.get("triage_level", "Unknown")
        tests = top.get("recommended_tests", "Not available")
        advice = top.get("advice", "Not available")

        lines = []
        lines.append("Here is a simple explanation of the result.\n")
        lines.append(f"Based on the symptoms entered, the system found a health insight that is most closely related to: {diagnosis}.")
        lines.append(f"The recommended next step is to consult a {specialty} specialist.")
        lines.append(f"The urgency level for this case is marked as {urgency}.")
        lines.append(f"Suggested tests include: {tests}.")
        lines.append(f"General advice from the support system is: {advice}.")

        if insurance_text:
            lines.append(f"\n{insurance_text}")
        if history_summary:
            lines.append(f"\nHistory note: {history_summary}")
            
        if doctors:
            lines.append("\nSuggested doctors based on the selected city and specialty include:")
            for doc in doctors[:3]:
                lines.append(f"- {doc.get('name', 'Unknown')} in {doc.get('city', 'Unknown')} ({doc.get('specialty', 'Unknown')}, rating {doc.get('rating', 'N/A')})")

        lines.append("\nThis explanation is for educational decision-support purposes only and is not a medical diagnosis.")
        lines.append("Please consult a licensed healthcare professional.")

        return "\n".join(lines)

    def _build_rewrite_prompt(self, structured_text: str) -> str:
        prompt = f"""Rewrite the following healthcare support explanation in simple, calm, patient-friendly English.
Rules:
- Do not add new medical facts.
- Do not remove important information.
- Do not use the word diagnosis. Keep the meaning unchanged.
- Keep it clear and reassuring.
- Keep it between 5 and 8 sentences.
- End with: Please consult a licensed healthcare professional.

Text:
{structured_text}"""
        return prompt.strip()

    def _generate_with_flan(self, prompt: str) -> str:
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("FLAN model is not loaded.")
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.2,
                do_sample=True,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.strip()

    def _is_bad_rewrite(self, text: str) -> bool:
        if not text:
            return True
        stripped = text.strip().lower()
        if len(stripped) < 80:
            return True
        bad_patterns = [
            "the doctor will", "a doctor may be able", "provide a brief", 
            "clear and reassuring answer", "the first step"
        ]
        if any(p in stripped for p in bad_patterns):
            return True
        return False

    def explain_model_only(self, payload: Dict[str, Any]) -> str:
        if not self.model_loaded:
            return f"MODEL NOT LOADED: {self.load_error}"
        structured_text = self._build_structured_explanation(payload)
        prompt = self._build_rewrite_prompt(structured_text)
        return self._generate_with_flan(prompt)

    def explain(self, payload: Dict[str, Any], use_ai: bool = False) -> str:
        structured_text = self._build_structured_explanation(payload)
        if not use_ai:
            return structured_text
        if not self.model_loaded:
            return structured_text
        try:
            rewritten = self.explain_model_only(payload)
            if self._is_bad_rewrite(rewritten):
                return structured_text
            return rewritten
        except Exception:
            return structured_text