from .reliable_engine import ReliableRiskEngine
from ml.reliable.preprocessing import normalize_sms


class SMSRiskEngine(ReliableRiskEngine):
    normalize_text = staticmethod(normalize_sms)

    def __init__(self):
        super().__init__("sms")
