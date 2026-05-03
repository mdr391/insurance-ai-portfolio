"""
Insurance-Domain Evaluation Datasets.

Ground-truth test cases curated by senior underwriters, claims specialists,
and compliance officers at Liberty Mutual.

Each case has:
  - question: What the system under test receives
  - expected_answer: Reference answer for comparison
  - expected_keywords: Must appear in a correct answer
  - forbidden_phrases: Must NOT appear (hallucination guards)
  - difficulty: easy / medium / hard
  - category: Sub-topic for granular reporting

In production, datasets are stored in a versioned dataset store
(DVC + Azure Blob Storage) and loaded at eval time.
New cases are added through a PR review process with subject-matter expert sign-off.
"""

from models.schemas import EvalCase, EvalDataset

# ── Underwriting Q&A Dataset ─────────────────────────────────────────────────
UNDERWRITING_QA_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What is the maximum single-location TIV for habitational risks without reinsurance approval?",
        expected_answer="The maximum single-location TIV for habitational risks is $50M without facultative reinsurance approval. Properties exceeding this threshold require a minimum 45-day lead time for reinsurance submission.",
        expected_keywords=["50m", "50 million", "reinsurance", "facultative", "45"],
        forbidden_phrases=["100m", "25m", "unlimited"],
        difficulty="easy",
        category="property_limits",
        requires_citation=True,
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="Which FEMA flood zones are eligible for our manuscript flood endorsement?",
        expected_answer="Only properties in FEMA Flood Zone X or Zone B are eligible for the manuscript flood endorsement. Zones A, AE, V, and VE are non-eligible.",
        expected_keywords=["zone x", "zone b", "eligible", "a", "ae", "v", "ve", "non-eligible"],
        forbidden_phrases=["zone a is eligible", "all zones"],
        difficulty="medium",
        category="flood_coverage",
        requires_citation=True,
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What experience modification factor threshold requires senior underwriter approval?",
        expected_answer="An experience modification factor (e-mod) above 1.25 requires senior underwriter approval and a mandatory risk improvement plan. E-mods above 1.50 are ineligible for the standard market.",
        expected_keywords=["1.25", "senior", "approval", "1.50", "ineligible"],
        forbidden_phrases=["1.0", "1.10", "no threshold"],
        difficulty="easy",
        category="workers_comp",
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What cyber security controls are mandatory before we can offer cyber liability coverage?",
        expected_answer="Mandatory controls include: MFA on all email and remote access, EDR/XDR endpoint protection on 100% of endpoints, immutable offsite backups tested quarterly, and critical patches applied within 30 days. Accounts lacking MFA are ineligible.",
        expected_keywords=["mfa", "multi-factor", "edr", "backup", "patch", "30 days", "ineligible"],
        forbidden_phrases=["optional", "no requirements", "any configuration"],
        difficulty="medium",
        category="cyber",
        requires_citation=True,
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="Is liquor liability included in a standard GL policy for a bar?",
        expected_answer="No. Liquor liability is excluded from standard GL for any insured whose operations involve selling or serving alcoholic beverages. A separate liquor liability (dram shop) policy is required for bars and restaurants.",
        expected_keywords=["excluded", "dram shop", "separate", "bars", "restaurants"],
        forbidden_phrases=["included in standard gl", "covered automatically"],
        difficulty="easy",
        category="liability",
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What is the maximum business interruption period available for standard commercial property accounts?",
        expected_answer="Standard accounts receive up to 12 months of gross profit for business interruption coverage. Extended BI up to 24 months may be offered to accounts with 3 or more years of loss-free history.",
        expected_keywords=["12 months", "24 months", "loss-free", "3 years", "gross profit"],
        forbidden_phrases=["36 months", "unlimited", "no maximum"],
        difficulty="medium",
        category="property_bi",
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What sprinkler requirements apply to warehouse risks over 10,000 square feet?",
        expected_answer="Sprinkler systems are mandatory for warehouses exceeding 10,000 square feet. Additionally, accounts with TIV above $5M must have a central station monitored fire alarm system. Failure to maintain these voids the protective safeguard warranty.",
        expected_keywords=["mandatory", "10,000", "sprinkler", "5m", "central station"],
        forbidden_phrases=["optional", "recommended only", "no requirement"],
        difficulty="medium",
        category="property_safeguards",
    ),
    EvalCase(
        dataset=EvalDataset.UNDERWRITING_QA,
        question="What ransomware sublimit applies to a healthcare organization with a $10M cyber policy?",
        expected_answer="Healthcare accounts are subject to a maximum ransomware sublimit of $5M, regardless of the overall policy limit. The retention for ransomware is 2x the standard cyber retention.",
        expected_keywords=["5m", "5 million", "healthcare", "sublimit", "2x", "retention"],
        forbidden_phrases=["10m", "full limit", "no sublimit"],
        difficulty="hard",
        category="cyber_ransomware",
        requires_citation=True,
    ),
]

# ── Claims Triage Dataset ─────────────────────────────────────────────────────
CLAIMS_TRIAGE_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.CLAIMS_TRIAGE,
        question="A claim is reported 45 days after a vehicle collision with no explanation for the delay. What fraud signals should be noted?",
        expected_answer="Late reporting (45 days, exceeding the 30-day threshold) is a medium-severity fraud signal. The claim should be flagged for enhanced documentation review. Additional signals to look for include inconsistent timelines, lack of witnesses, and prior claims history.",
        expected_keywords=["late reporting", "30 days", "45 days", "flag", "enhanced", "medium"],
        forbidden_phrases=["no concern", "normal", "approve immediately"],
        difficulty="medium",
        category="fraud_signals",
    ),
    EvalCase(
        dataset=EvalDataset.CLAIMS_TRIAGE,
        question="A claimant reports a total loss auto claim for $45,000 with no witnesses and an attorney already retained. How should this be routed?",
        expected_answer="This claim exhibits multiple high-severity fraud signals: high claim value ($45,000), no witnesses, and attorney involvement. It should be routed to the Fraud Investigation Unit with urgent priority. Payment should be suspended pending review.",
        expected_keywords=["fraud investigation", "urgent", "suspend", "attorney", "no witnesses"],
        forbidden_phrases=["standard review", "fast track", "approve"],
        difficulty="hard",
        category="routing",
    ),
    EvalCase(
        dataset=EvalDataset.CLAIMS_TRIAGE,
        question="What claim amount patterns should trigger a threshold avoidance flag?",
        expected_answer="Claim amounts just below reporting thresholds — specifically between $9,000-$9,999 or $19,000-$19,999 — should be flagged for threshold avoidance review. These amounts are suspicious because they may be artificially set to avoid mandatory reporting.",
        expected_keywords=["9,000", "9,999", "19,000", "threshold", "avoidance", "suspicious"],
        forbidden_phrases=["no concern", "round numbers are fine"],
        difficulty="hard",
        category="fraud_patterns",
    ),
]

# ── Coverage Reasoning Dataset ────────────────────────────────────────────────
COVERAGE_REASONING_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.COVERAGE_REASONING,
        question="A contractor performs roofing work that is completed on June 1. A property damage claim from that work is reported on September 15. Is this covered?",
        expected_answer="Yes. Products-completed operations (PCO) coverage applies to property damage occurring away from the insured's premises after the work is complete. The claim is within the policy period and PCO coverage applies to post-completion incidents.",
        expected_keywords=["products-completed operations", "pco", "covered", "post-completion", "policy period"],
        forbidden_phrases=["not covered", "excluded", "work is done"],
        difficulty="hard",
        category="gl_coverage",
    ),
    EvalCase(
        dataset=EvalDataset.COVERAGE_REASONING,
        question="An office company holds a holiday party where alcohol is served. An attendee is injured. Does standard GL cover this?",
        expected_answer="Yes. Host liquor liability is included in standard GL for non-habitually-serving insureds such as an office holiday party, up to $1M per occurrence. This is distinct from operations where alcohol service is the primary business activity.",
        expected_keywords=["host liquor", "included", "1m", "non-habitually", "holiday party"],
        forbidden_phrases=["excluded", "requires separate policy", "not covered"],
        difficulty="medium",
        category="liquor_liability",
    ),
]

# ── Fraud Signals Dataset ─────────────────────────────────────────────────────
FRAUD_SIGNALS_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.FRAUD_SIGNALS,
        question="List the top 5 fraud indicators for staged auto accident claims.",
        expected_answer="Top indicators include: (1) late reporting beyond 30 days, (2) no witnesses or witnesses who are known associates, (3) attorney retained immediately at first contact, (4) inconsistent or self-correcting narrative, (5) prior claims history with similar patterns.",
        expected_keywords=["late reporting", "no witnesses", "attorney", "inconsistent", "prior claims"],
        forbidden_phrases=["no fraud indicators", "impossible to detect"],
        difficulty="medium",
        category="auto_fraud",
    ),
    EvalCase(
        dataset=EvalDataset.FRAUD_SIGNALS,
        question="A claim amount is exactly $9,999. Why is this suspicious and what should an adjuster do?",
        expected_answer="A claim of exactly $9,999 may represent threshold avoidance — intentionally staying below a $10,000 reporting threshold. The adjuster should flag this as a medium-severity fraud signal, request full documentation, and note it in the claim file for supervisor review.",
        expected_keywords=["threshold avoidance", "10,000", "flag", "documentation", "supervisor"],
        forbidden_phrases=["coincidence", "normal amount", "no concern"],
        difficulty="medium",
        category="threshold_avoidance",
    ),
]

# ── Policy Extraction Dataset ─────────────────────────────────────────────────
POLICY_EXTRACTION_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.POLICY_EXTRACTION,
        question="Extract the key coverage differences between a standard and enhanced BI endorsement.",
        expected_answer="Standard BI covers up to 12 months of gross profit. Enhanced BI extends coverage to 24 months and is available to accounts with 3+ years of loss-free history at additional premium.",
        expected_keywords=["12 months", "24 months", "3 years", "loss-free", "gross profit"],
        forbidden_phrases=["no difference", "same coverage"],
        difficulty="medium",
        category="bi_comparison",
    ),
]

# ── Regulatory Compliance Dataset ─────────────────────────────────────────────
REGULATORY_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.REGULATORY_COMPLIANCE,
        question="Can an underwriter provide a verbal commitment to bind coverage before the policy is issued?",
        expected_answer="No. Verbal commitments to bind can create legal obligations. All coverage commitments must be documented in writing with appropriate conditions. Underwriters should avoid any language that could be construed as a binding commitment before formal policy issuance.",
        expected_keywords=["no", "verbal", "written", "documented", "binding", "not recommended"],
        forbidden_phrases=["yes", "verbal is fine", "informal commitment is acceptable"],
        difficulty="medium",
        category="binding_authority",
    ),
    EvalCase(
        dataset=EvalDataset.REGULATORY_COMPLIANCE,
        question="Is it permissible to use race or national origin as an underwriting factor?",
        expected_answer="No. Using race, national origin, religion, or other protected characteristics as underwriting factors is prohibited under the Fair Housing Act, Equal Credit Opportunity Act, and state insurance regulations. Doing so constitutes illegal discrimination.",
        expected_keywords=["no", "prohibited", "illegal", "discrimination", "fair housing", "protected"],
        forbidden_phrases=["yes", "permissible", "allowed", "can be considered"],
        difficulty="easy",
        category="fair_lending",
    ),
]

# ── Customer Communications Dataset ──────────────────────────────────────────
CUSTOMER_COMMS_CASES: list[EvalCase] = [
    EvalCase(
        dataset=EvalDataset.CUSTOMER_COMMS,
        question="Write a brief explanation of a flood exclusion for a policyholder who is confused about why their water damage claim was denied.",
        expected_answer="The response should clearly explain the flood exclusion in plain language, acknowledge the customer's frustration, explain what flood coverage requires (separate policy), and outline next steps without making legal commitments.",
        expected_keywords=["flood exclusion", "separate", "coverage", "apologize", "next steps"],
        forbidden_phrases=["your claim is approved", "we made an error", "guaranteed coverage"],
        difficulty="medium",
        category="claims_communication",
    ),
]

# ── Master dataset registry ────────────────────────────────────────────────────
ALL_DATASETS: dict[EvalDataset, list[EvalCase]] = {
    EvalDataset.UNDERWRITING_QA:       UNDERWRITING_QA_CASES,
    EvalDataset.CLAIMS_TRIAGE:         CLAIMS_TRIAGE_CASES,
    EvalDataset.COVERAGE_REASONING:    COVERAGE_REASONING_CASES,
    EvalDataset.FRAUD_SIGNALS:         FRAUD_SIGNALS_CASES,
    EvalDataset.POLICY_EXTRACTION:     POLICY_EXTRACTION_CASES,
    EvalDataset.REGULATORY_COMPLIANCE: REGULATORY_CASES,
    EvalDataset.CUSTOMER_COMMS:        CUSTOMER_COMMS_CASES,
}


def load_dataset(dataset: EvalDataset, sample_size: int | None = None) -> list[EvalCase]:
    """Load cases for a dataset, optionally sampling."""
    cases = ALL_DATASETS.get(dataset, [])
    if sample_size and sample_size < len(cases):
        import random
        return random.sample(cases, sample_size)
    return cases


def load_all_cases(
    datasets: list[EvalDataset] | None = None,
    sample_size: int | None = None,
) -> list[EvalCase]:
    """Load cases from one or more datasets."""
    target = datasets or list(EvalDataset)
    cases = []
    for ds in target:
        cases.extend(load_dataset(ds, sample_size))
    return cases


def dataset_stats() -> dict:
    """Return dataset statistics."""
    return {
        ds.value: {
            "count": len(cases),
            "difficulties": {d: sum(1 for c in cases if c.difficulty == d) for d in ["easy", "medium", "hard"]},
            "categories": list({c.category for c in cases}),
            "requires_citation": sum(1 for c in cases if c.requires_citation),
        }
        for ds, cases in ALL_DATASETS.items()
    }
