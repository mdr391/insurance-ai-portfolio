"""
PII Detection & Redaction Service.

All prompts are scrubbed BEFORE being forwarded to any LLM provider.
Redaction is applied to both inbound prompts and outbound completions.

Production stack:
  Microsoft Presidio + spaCy NER + custom insurance recognizers

  from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
  from presidio_anonymizer import AnonymizerEngine
  from presidio_analyzer.nlp_engine import NlpEngineProvider

Custom recognizers for insurance domain:
  - Policy numbers (LM-XXXXXXX, POL-XXXXXXX)
  - Claim numbers (CLM-XXXXXXX)
  - VIN numbers (17-char alphanumeric)
  - NAIC codes
  - Social Security Numbers
  - Driver license numbers
  - Medical record numbers

All redaction events are counted and logged in the audit trail.
Redaction is reversible in a secured vault for authorized reviewers.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    original_hash: str          # SHA-256 of original (for audit)
    redacted_text: str          # Safe text forwarded to LLM
    tokens_redacted: int
    entity_types_found: list[str]
    was_redacted: bool


# Insurance-domain PII patterns
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Entity type, compiled pattern, replacement token
    (re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),                    "SSN",            "[SSN-REDACTED]"),
    (re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b'),                             "VIN",            "[VIN-REDACTED]"),
    (re.compile(r'\b(?:LM|POL|CLM|UW)[-\s]?\d{6,10}\b', re.IGNORECASE), "POLICY_NUM",     "[POLICY-NUM-REDACTED]"),
    (re.compile(r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),"PHONE",          "[PHONE-REDACTED]"),
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),"EMAIL",         "[EMAIL-REDACTED]"),
    (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),                         "CREDIT_CARD",    "[CARD-REDACTED]"),
    (re.compile(r'\b\d{1,5}\s+\w+\s+(?:St|Ave|Blvd|Dr|Rd|Ln|Way|Ct|Pl|Street|Avenue|Road)\b', re.IGNORECASE), "ADDRESS", "[ADDRESS-REDACTED]"),
    (re.compile(r'\b(?:DOB|Date of Birth|born)[:\s]+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', re.IGNORECASE), "DOB", "[DOB-REDACTED]"),
    (re.compile(r'\b[A-Z]{1,2}\d{6,8}\b'),                               "DL_NUMBER",      "[DL-REDACTED]"),
    (re.compile(r'\bMRN[-:\s]?\d{6,10}\b', re.IGNORECASE),               "MEDICAL_RECORD", "[MRN-REDACTED]"),
    # Named person detection (simplified — Presidio uses NER in production)
    (re.compile(r'(?:claimant|insured|adjuster|policyholder)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)', re.IGNORECASE), "PERSON_NAME", "[NAME-REDACTED]"),
]

# Patterns for output response scanning (LLM may reconstruct PII)
_OUTPUT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'), "[SSN-REDACTED]"),
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "[EMAIL-REDACTED]"),
    (re.compile(r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[PHONE-REDACTED]"),
]


class PIIService:
    """
    Bidirectional PII redaction — applied to both prompts and completions.

    Design principle: the LLM provider (Azure OpenAI, Anthropic, etc.) should
    NEVER receive raw PII. This protects the enterprise from:
      1. PII appearing in training data (if provider fine-tunes on logs)
      2. Regulatory exposure (GLBA, HIPAA, CCPA)
      3. Audit trail contamination

    In production:
        presidio_analyzer   → NER-based entity detection (spaCy en_core_web_lg)
        presidio_anonymizer → Operator-based replacement with reversible encryption
        Custom vault        → Authorized investigators can de-anonymize if needed
    """

    def redact_prompt(self, text: str) -> RedactionResult:
        """Redact PII from user prompt before forwarding to LLM."""
        original_hash = hashlib.sha256(text.encode()).hexdigest()
        redacted = text
        found_types: list[str] = []
        count = 0

        for pattern, entity_type, replacement in _PATTERNS:
            matches = pattern.findall(redacted)
            if matches:
                found_types.append(entity_type)
                count += len(matches)
                redacted = pattern.sub(replacement, redacted)

        if count > 0:
            logger.info(f"PII redaction: {count} tokens across {len(found_types)} entity types: {found_types}")

        return RedactionResult(
            original_hash=original_hash,
            redacted_text=redacted,
            tokens_redacted=count,
            entity_types_found=found_types,
            was_redacted=count > 0,
        )

    def scan_completion(self, text: str) -> tuple[str, int]:
        """
        Scan LLM completion for reconstructed PII.
        The LLM may occasionally reconstruct PII from context — catch it on the way out.
        Returns (cleaned_text, redacted_count).
        """
        cleaned = text
        count = 0
        for pattern, replacement in _OUTPUT_PATTERNS:
            matches = pattern.findall(cleaned)
            if matches:
                count += len(matches)
                cleaned = pattern.sub(replacement, cleaned)
        if count > 0:
            logger.warning(f"OUTPUT SCAN: {count} PII tokens found in LLM completion — redacted before delivery")
        return cleaned, count

    def hash_for_audit(self, value: str) -> str:
        """Deterministic hash for audit logging (not reversible)."""
        return hashlib.sha256(f"pii-audit-salt:{value}".encode()).hexdigest()


pii_service = PIIService()
