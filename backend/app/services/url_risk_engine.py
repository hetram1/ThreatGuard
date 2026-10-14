from .reliable_engine import ReliableRiskEngine
from ml.reliable.preprocessing import normalize_url


class URLRiskEngine(ReliableRiskEngine):
    normalize_url = staticmethod(normalize_url)

    def __init__(self):
        super().__init__("url")
