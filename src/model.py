import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.severity_scoring import format_prediction


y_pred_label = "DDoS"
confidence = 0.93

result = format_prediction(y_pred_label, confidence)
print(result)
# {'label': 'DDoS', 'confidence': 0.93, 'severity_score': 9.3}
