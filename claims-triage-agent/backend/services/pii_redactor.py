"""
PII Redaction Service.

In production this uses Microsoft Presidio (pip install presidio-analyzer presidio-anonymizer)
with custom insurance-domain recognizers for policy numbers, claim IDs, VINs, etc.

This implementation shows the same interface with regex-based redaction
so the demo runs without heavy NLP dependencies.

Production usage:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    redacted = anonymizer.anonymize(text=text, analyzer_results=results)
"""

import re
import logging

logger = logging.getLogger(__name__)

# PII patterns for insurance domain
PII_PATTERNS = [
    # Names (simplified — Presidio uses NER in production)
    (r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', '[NAME]'),
    # Phone numbers
    (r'\b(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})\b', '[PHONE]'),
    # SSN
    (r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', '[SSN]'),
    # Email
    (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', '[EMAIL]'),
    # Credit card
    (r'\b(?:\d{4}[\s\-]?){3}\d{4}\b', '[CARD_NUMBER]'),
    # Date of birth patterns
    (r'\bDOB:?\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', '[DOB]'),
    # VIN (full)
    (r'\b[A-HJ-NPR-Z0-9]{17}\b', '[VIN]'),
    # Policy number (common formats)
    (r'\b[A-Z]{2,3}[-\s]?\d{6,10}\b', '[POLICY_NUMBER]'),
    # Driver's license
    (r'\b[A-Z]{1,2}\d{6,8}\b', '[DL_NUMBER]'),
    # Street addresses
    (r'\b\d{1,5}\s+([A-Z][a-z]+\s+){1,3}(St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl)\.?\b', '[ADDRESS]'),
]


class PIIRedactor:
    """
    Redacts PII from claim text before it enters the LLM pipeline.
    
    Critical for HIPAA/GLBA compliance and enterprise AI governance.
    Every LLM call in this pipeline operates on redacted text only.
    """

    def __init__(self):
        self.last_redaction_count = 0
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in PII_PATTERNS
        ]

    def redact(self, text: str) -> str:
        """Apply all PII redaction patterns. Returns redacted text."""
        redacted = text
        count = 0
        for pattern, replacement in self._compiled:
            matches = pattern.findall(redacted)
            if matches:
                count += len(matches)
                redacted = pattern.sub(replacement, redacted)
        self.last_redaction_count = count
        logger.debug(f"PII redaction: {count} tokens redacted")
        return redacted

    def redact_structured(self, data: dict) -> dict:
        """
        Recursively redact PII from a dict (for logging structured claim data).
        Sensitive fields are replaced with hashed values, not redacted tokens.
        """
        sensitive_keys = {
            'claimant_name', 'contact_email', 'contact_phone',
            'ssn', 'date_of_birth', 'address', 'policy_number'
        }
        import hashlib
        result = {}
        for k, v in data.items():
            if k in sensitive_keys and isinstance(v, str):
                result[k] = '[REDACTED:' + hashlib.sha256(v.encode()).hexdigest()[:8] + ']'
            elif isinstance(v, dict):
                result[k] = self.redact_structured(v)
            else:
                result[k] = v
        return result
