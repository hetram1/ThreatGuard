import re
import numpy as np
import pandas as pd


URL_PATTERN = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|net|org|in|co|info|xyz|site|online|top|biz)\b)",
    re.IGNORECASE,
)

IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"
)

MONEY_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr|usd|\$|£|€|\b\d+(?:,\d{2,3})*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|rupees))",
    re.IGNORECASE,
)

OTP_PATTERN = re.compile(
    r"\b(?:otp|one[\s-]?time[\s-]?password|verification[\s-]?code|passcode)\b",
    re.IGNORECASE,
)

URGENCY_PATTERN = re.compile(
    r"\b(?:urgent|urgently|immediately|now|today|tonight|"
    r"asap|hurry|quick|quickly|expire|expired|deadline|"
    r"within\s+\d+\s*(?:hour|hours|minute|minutes)|"
    r"turant|abhi|jaldi|aaj|aaj\s+raat)\b",
    re.IGNORECASE,
)

KYC_PATTERN = re.compile(
    r"\b(?:kyc|know\s+your\s+customer|pan|aadhaar|aadhar|"
    r"verification|verify|verified|verification)\b",
    re.IGNORECASE,
)

BANK_PATTERN = re.compile(
    r"\b(?:bank|account|sbi|hdfc|icici|axis|kotak|"
    r"credit\s+card|debit\s+card|atm|netbanking|"
    r"net\s+banking|wallet)\b",
    re.IGNORECASE,
)

PAYMENT_PATTERN = re.compile(
    r"\b(?:upi|payment|pay|paid|refund|cashback|"
    r"transaction|transfer|merchant|fee|processing\s+fee|"
    r"recharge|bill)\b",
    re.IGNORECASE,
)

THREAT_PATTERN = re.compile(
    r"\b(?:blocked|block|freeze|frozen|suspend|suspended|"
    r"disconnect|disconnected|closed|closure|locked|"
    r"compromised|fraud|unauthorized|unauthorised)\b",
    re.IGNORECASE,
)

REWARD_PATTERN = re.compile(
    r"\b(?:won|winner|win|prize|reward|lottery|"
    r"lucky\s+draw|congratulations|selected|"
    r"cashback|bonus|free|gift)\b",
    re.IGNORECASE,
)

SOCIAL_ENGINEERING_PATTERN = re.compile(
    r"\b(?:claim|click|verify|update|confirm|"
    r"send|share|provide|submit|call|contact|"
    r"register|activate|reactivate)\b",
    re.IGNORECASE,
)

OBFUSCATION_PATTERN = re.compile(
    r"[01345789@$]",
)


FEATURE_NAMES = [
    "text_length",
    "word_count",
    "avg_word_length",
    "digit_count",
    "digit_ratio",
    "uppercase_count",
    "uppercase_ratio",
    "special_char_count",
    "special_char_ratio",
    "exclamation_count",
    "question_count",
    "url_count",
    "has_url",
    "has_ip_address",
    "phone_like_count",
    "has_phone",
    "money_count",
    "has_money",
    "otp_count",
    "has_otp",
    "urgency_count",
    "has_urgency",
    "kyc_count",
    "has_kyc",
    "bank_count",
    "has_bank",
    "payment_count",
    "has_payment",
    "threat_count",
    "has_threat",
    "reward_count",
    "has_reward",
    "social_engineering_count",
    "has_social_engineering",
    "obfuscation_count",
    "has_obfuscation",
    "url_length_max",
    "url_has_https",
]


def _count(pattern, text):
    return len(
        pattern.findall(text)
    )


def extract_sms_features(
    texts
):
    rows = []

    for raw_text in texts:

        text = str(raw_text)

        lower = text.lower()

        words = re.findall(
            r"\b\w+\b",
            text,
            flags=re.UNICODE,
        )

        word_count = len(words)

        text_length = len(text)

        digit_count = sum(
            char.isdigit()
            for char in text
        )

        uppercase_count = sum(
            char.isupper()
            for char in text
        )

        special_char_count = sum(
            not char.isalnum()
            and not char.isspace()
            for char in text
        )

        urls = URL_PATTERN.findall(
            text
        )

        url_lengths = [
            len(url)
            for url in urls
        ]

        url_https = any(
            url.lower().startswith(
                "https://"
            )
            for url in urls
        )

        phone_count = _count(
            PHONE_PATTERN,
            text,
        )

        money_count = _count(
            MONEY_PATTERN,
            text,
        )

        otp_count = _count(
            OTP_PATTERN,
            text,
        )

        urgency_count = _count(
            URGENCY_PATTERN,
            text,
        )

        kyc_count = _count(
            KYC_PATTERN,
            text,
        )

        bank_count = _count(
            BANK_PATTERN,
            text,
        )

        payment_count = _count(
            PAYMENT_PATTERN,
            text,
        )

        threat_count = _count(
            THREAT_PATTERN,
            text,
        )

        reward_count = _count(
            REWARD_PATTERN,
            text,
        )

        social_count = _count(
            SOCIAL_ENGINEERING_PATTERN,
            text,
        )

        obfuscation_count = _count(
            OBFUSCATION_PATTERN,
            text,
        )

        rows.append(
            [
                text_length,

                word_count,

                (
                    np.mean(
                        [
                            len(word)
                            for word in words
                        ]
                    )
                    if words
                    else 0.0
                ),

                digit_count,

                (
                    digit_count
                    / max(
                        text_length,
                        1
                    )
                ),

                uppercase_count,

                (
                    uppercase_count
                    / max(
                        text_length,
                        1
                    )
                ),

                special_char_count,

                (
                    special_char_count
                    / max(
                        text_length,
                        1
                    )
                ),

                text.count("!"),

                text.count("?"),

                len(urls),

                int(
                    len(urls) > 0
                ),

                int(
                    bool(
                        IP_PATTERN.search(
                            text
                        )
                    )
                ),

                phone_count,

                int(
                    phone_count > 0
                ),

                money_count,

                int(
                    money_count > 0
                ),

                otp_count,

                int(
                    otp_count > 0
                ),

                urgency_count,

                int(
                    urgency_count > 0
                ),

                kyc_count,

                int(
                    kyc_count > 0
                ),

                bank_count,

                int(
                    bank_count > 0
                ),

                payment_count,

                int(
                    payment_count > 0
                ),

                threat_count,

                int(
                    threat_count > 0
                ),

                reward_count,

                int(
                    reward_count > 0
                ),

                social_count,

                int(
                    social_count > 0
                ),

                obfuscation_count,

                int(
                    obfuscation_count > 0
                ),

                max(
                    url_lengths,
                    default=0,
                ),

                int(
                    url_https
                ),
            ]
        )

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
    )
