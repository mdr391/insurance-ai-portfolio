"""
RAG Retrieval & Generation Engine — Underwriting Copilot.

Two-stage retrieval pipeline:
  Stage 1: Dense vector retrieval (Azure AI Search HNSW)
  Stage 2: Cross-encoder reranking (Cohere Rerank / BGE-reranker)

Generation:
  - GPT-4o with citation-aware system prompt
  - Structured output: answer + inline citation markers [1], [2]...
  - Confidence scoring based on retrieval scores + answer grounding
  - Source attribution for every factual claim

Production stack:
  - LlamaIndex RetrieverQueryEngine
  - Azure AI Search hybrid retrieval (BM25 + vector)
  - Cohere Rerank v3 or BGE-reranker-large
  - Azure OpenAI GPT-4o (with JSON mode for structured output)
  - RAGAS for evaluation
"""

import json
import logging
import re
import time
import uuid
from typing import Optional

from models.schemas import (
    Citation, CopilotQuery, CopilotResponse,
    DocumentChunk, DocumentType, RetrievedChunk,
)

logger = logging.getLogger(__name__)

# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert underwriting assistant at a large commercial insurer.
You answer underwriters' questions using ONLY the provided source excerpts from underwriting guidelines, coverage manuals, and loss run reports.

CRITICAL RULES:
1. Base every factual claim on the provided sources. Never invent coverage rules, limits, or exclusions.
2. Mark every factual claim with an inline citation [N] referencing the source number.
3. If sources are insufficient to fully answer the question, explicitly say so and indicate what additional information would be needed.
4. Use professional insurance terminology. Be precise about coverage terms, limits, sublimits, and exclusions.
5. Flag any apparent contradictions between sources.

Respond in this JSON format:
{
  "answer": "Your answer with inline citations [1], [2]...",
  "confidence": "high" | "medium" | "low",
  "confidence_rationale": "Brief explanation of confidence level",
  "gaps": "Any gaps in the source material that limit completeness"
}"""

RETRIEVAL_PROMPT_TEMPLATE = """Underwriter question: {query}

Source excerpts:
{sources}

Answer the question using only these sources. Cite every fact."""


class UnderwritingRAGEngine:
    """
    End-to-end RAG pipeline for underwriting queries.

    Production usage with LlamaIndex:
        from llama_index.core import VectorStoreIndex, QueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        retriever = VectorIndexRetriever(index=index, similarity_top_k=15)
        reranker = CohereRerank(api_key=..., top_n=5)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker],
            response_synthesizer=get_response_synthesizer(
                response_mode="compact",
                llm=AzureOpenAI(model="gpt-4o", ...),
            ),
        )
        response = query_engine.query(question)
    """

    # Simulated document corpus — in production these come from Azure AI Search
    CORPUS: list[dict] = [
        {
            "doc_id": "UG-2024-001",
            "doc_title": "Commercial Property Underwriting Guidelines v12.3",
            "doc_type": "underwriting_guideline",
            "section": "Coverage Limits & Sublimits",
            "page": 34,
            "content": "Maximum single-location TIV for habitational risks is $50M without reinsurance approval. Properties exceeding this threshold require submission to facultative reinsurance with a minimum 45-day lead time. Business interruption coverage is capped at 12 months of gross profit for standard accounts; extended BI up to 24 months may be offered to qualifying accounts with 3+ years of loss-free history.",
        },
        {
            "doc_id": "UG-2024-001",
            "doc_title": "Commercial Property Underwriting Guidelines v12.3",
            "doc_type": "underwriting_guideline",
            "section": "Flood & Water Exclusions",
            "page": 67,
            "content": "Standard commercial property policies exclude flood as defined in ISO CP 10 30. Flood coverage may be offered as a manuscript endorsement for properties in FEMA Flood Zone X or B only; Zones A, AE, V, and VE are non-eligible. The maximum flood sublimit is $5M per occurrence. Coastal properties within 1,000 feet of tidal water require a separate wind/hail deductible of 3-5% of TIV. Zone eligibility applies to all commercial property flood endorsements.",
        },
        {
            "doc_id": "UG-2024-001",
            "doc_title": "Commercial Property Underwriting Guidelines v12.3",
            "doc_type": "underwriting_guideline",
            "section": "Protective Safeguards Requirements",
            "page": 89,
            "content": "All accounts with TIV above $5M must have a central station monitored fire alarm system. Sprinkler systems are mandatory for: (a) restaurants and food service operations, (b) warehouses exceeding 10,000 sq ft, (c) any occupancy with woodworking operations. Failure to maintain required safeguards voids the protective safeguard warranty and may result in mid-term cancellation.",
        },
        {
            "doc_id": "LR-2024-Q2",
            "doc_title": "Commercial Auto Loss Run Analysis Q2 2024",
            "doc_type": "loss_run",
            "section": "Summary Statistics",
            "page": 3,
            "content": "Q2 2024 commercial auto book: 1,247 policies in force, 89 reported claims, 7.13% claim frequency. Average severity: $18,400. Total incurred losses: $1.638M. Combined ratio: 94.2%. Top loss drivers: rear-end collisions (34%), cargo damage (21%), driver negligence (18%). Accounts with 3+ at-fault claims in 36 months should be referred to senior underwriting for renewal evaluation.",
        },
        {
            "doc_id": "LR-2024-Q2",
            "doc_title": "Commercial Auto Loss Run Analysis Q2 2024",
            "doc_type": "loss_run",
            "section": "Driver Risk Segmentation",
            "page": 11,
            "content": "Driver risk tiers: Tier 1 (preferred) requires MVR clear of major violations for 5 years and <2 minor violations in 3 years. Tier 2 (standard) allows 1 major violation or 3 minor violations in 3 years. Tier 3 (non-standard) requires senior approval and a 25% surcharge on base premium. Drivers with DUI/DWI in the past 7 years are ineligible. CDL holders are evaluated separately under fleet program guidelines.",
        },
        {
            "doc_id": "CM-2024-003",
            "doc_title": "General Liability Coverage Manual 2024",
            "doc_type": "coverage_manual",
            "section": "Products-Completed Operations",
            "page": 22,
            "content": "Products-completed operations (PCO) coverage applies to bodily injury or property damage occurring away from the insured's premises after the work is complete or the product relinquished. Standard PCO limits mirror the GL occurrence limit. Extended PCO tail coverage of up to 5 years post-policy expiration is available for contractors at an additional 15-25% of GL premium. Medical device manufacturers require a separate PCO manuscript policy.",
        },
        {
            "doc_id": "CM-2024-003",
            "doc_title": "General Liability Coverage Manual 2024",
            "doc_type": "coverage_manual",
            "section": "Contractual Liability",
            "page": 38,
            "content": "Contractual liability coverage is included for liability assumed under an 'insured contract' as defined in CG 00 01. Blanket additional insured status is granted automatically when required by written contract. Additional insured coverage is limited to vicarious liability only; direct acts of the additional insured are excluded. The waiver of subrogation endorsement (CG 24 04) may be added at underwriter's discretion for accounts with gross premium above $50,000.",
        },
        {
            "doc_id": "CM-2024-003",
            "doc_title": "General Liability Coverage Manual 2024",
            "doc_type": "coverage_manual",
            "section": "Liquor Liability",
            "page": 55,
            "content": "Liquor liability is excluded from standard GL for any insured whose operations involve manufacturing, selling, distributing, or serving alcoholic beverages. A separate liquor liability policy (dram shop) is required for bars, restaurants, and event venues serving alcohol. Host liquor liability is included in standard GL for non-habitually-serving insureds (e.g., office holiday parties) up to $1M per occurrence.",
        },
        {
            "doc_id": "RF-2024-001",
            "doc_title": "Workers Compensation Rate Filing 2024",
            "doc_type": "rate_filing",
            "section": "Experience Modification Factor",
            "page": 7,
            "content": "Experience modification factor (EMF or e-mod) is calculated by the NCCI or applicable state bureau using 3 prior policy years, excluding the most recent year. An e-mod above 1.25 requires senior underwriter approval and mandatory risk improvement plan. Accounts with e-mod above 1.50 are ineligible for standard market and must be referred to specialty or excess & surplus lines. E-mod calculation is mandatory for accounts with payroll exceeding $25,000 in any state.",
        },
        {
            "doc_id": "RF-2024-001",
            "doc_title": "Workers Compensation Rate Filing 2024",
            "doc_type": "rate_filing",
            "section": "Classification Codes",
            "page": 14,
            "content": "Clerical office employees (NCCI code 8810) carry a base rate of $0.42/$100 payroll. Contractors performing roof work (code 5551) carry $18.73/$100 payroll. Restaurants (code 9082) carry $3.18/$100 payroll. Misclassification of payroll is a material misrepresentation and grounds for policy rescission. Annual payroll audits are required for all WC accounts; failure to cooperate with audit results in estimated premium at 150% of expiring.",
        },
        {
            "doc_id": "UG-2024-002",
            "doc_title": "Cyber Liability Underwriting Guidelines 2024",
            "doc_type": "underwriting_guideline",
            "section": "Technology Requirements",
            "page": 12,
            "content": "All cyber applicants must complete the technology questionnaire (TQ-CY-2024). Minimum qualifying controls: (1) Multi-factor authentication on all email and remote access; (2) EDR/XDR endpoint protection on 100% of endpoints; (3) Immutable offsite backups tested at least quarterly; (4) Patch management with critical patches applied within 30 days. Accounts lacking MFA are ineligible for cyber coverage.",
        },
        {
            "doc_id": "UG-2024-002",
            "doc_title": "Cyber Liability Underwriting Guidelines 2024",
            "doc_type": "underwriting_guideline",
            "section": "Ransomware Sublimits",
            "page": 28,
            "content": "Ransomware coverage sublimits apply for accounts in higher-risk sectors: healthcare ($5M maximum sublimit), financial services ($5M), manufacturing ($10M), retail ($10M). All other sectors receive full policy limit for ransomware. Ransomware-specific retentions are 2x the standard cyber retention. Accounts that experienced a ransomware event in the prior 24 months require a 5-year minimum waiting period before ransomware sublimit applies at full policy limit.",
        },
    ]

    def __init__(self):
        self._index_corpus()

    def _index_corpus(self):
        """In production: this connects to Azure AI Search index."""
        self._chunks = [
            DocumentChunk(
                doc_id=c["doc_id"],
                doc_title=c["doc_title"],
                doc_type=DocumentType(c["doc_type"]),
                section=c.get("section"),
                page_number=c.get("page"),
                content=c["content"],
                token_count=len(c["content"].split()),
            )
            for c in self.CORPUS
        ]
        logger.info(f"RAG engine initialized with {len(self._chunks)} corpus chunks")

    async def query(self, copilot_query: CopilotQuery) -> CopilotResponse:
        """End-to-end RAG pipeline: retrieve → rerank → generate."""
        t_start = time.time()
        query_text = copilot_query.query_text

        # Stage 1: Dense vector retrieval
        t_retrieve = time.time()
        retrieved = await self._retrieve(query_text, copilot_query.top_k, copilot_query.doc_type_filter)
        retrieval_ms = int((time.time() - t_retrieve) * 1000)

        # Stage 2: Rerank
        reranked = await self._rerank(query_text, retrieved)

        # Stage 3: Generate grounded answer
        t_gen = time.time()
        generation = await self._generate(query_text, reranked)
        generation_ms = int((time.time() - t_gen) * 1000)

        total_ms = max(1, int((time.time() - t_start) * 1000))

        # Build citations
        citations = self._build_citations(reranked)

        return CopilotResponse(
            query_id=copilot_query.query_id,
            query_text=query_text,
            answer=generation["answer"],
            citations=citations,
            confidence_level=generation["confidence"],
            confidence_rationale=generation["confidence_rationale"],
            retrieved_chunks=len(retrieved),
            reranked_chunks=len(reranked),
            answer_tokens=generation["tokens"],
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
            total_latency_ms=total_ms,
            model_used="gpt-4o",
            embedding_model="text-embedding-3-large",
            cost_usd=generation["cost"],
        )

    async def _retrieve(
        self,
        query: str,
        top_k: int,
        doc_type_filter: Optional[list] = None,
    ) -> list[RetrievedChunk]:
        """
        Dense vector + BM25 hybrid retrieval from Azure AI Search.

        Production code (LlamaIndex):
            from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
            retriever = VectorIndexRetriever(index=az_index, similarity_top_k=top_k)
            nodes = retriever.retrieve(query)

        Direct Azure AI Search hybrid search:
            results = search_client.search(
                search_text=query,           # BM25 full-text
                vector_queries=[VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=top_k,
                    fields="embedding",
                )],
                filter=f"doc_type eq '{doc_type}'" if doc_type_filter else None,
                top=top_k,
                select=["chunk_id", "content", "doc_title", "section", "page_number"],
            )
        """
        # Simulate semantic similarity scoring
        import math
        query_words = set(query.lower().split())

        scored = []
        for chunk in self._chunks:
            if doc_type_filter and chunk.doc_type not in doc_type_filter:
                continue
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            jaccard = overlap / max(len(query_words | chunk_words), 1)
            # Boost with keyword matches on section/title
            section_boost = 0.1 if chunk.section and any(w in (chunk.section or "").lower() for w in query_words) else 0
            score = min(1.0, jaccard * 3.5 + section_boost + 0.05)
            scored.append(RetrievedChunk(chunk=chunk, similarity_score=round(score, 4)))

        scored.sort(key=lambda x: x.similarity_score, reverse=True)
        return scored[:top_k * 2]  # retrieve extra for reranking

    async def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Cross-encoder reranking for better precision.

        Production code (Cohere):
            from cohere import Client
            co = Client(api_key=os.getenv("COHERE_API_KEY"))
            results = co.rerank(
                query=query,
                documents=[c.chunk.content for c in candidates],
                model="rerank-english-v3.0",
                top_n=5,
            )
            reranked = [candidates[r.index] for r in results.results]
            for i, r in enumerate(results.results):
                reranked[i].rerank_score = r.relevance_score

        Or with sentence-transformers (local):
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            scores = model.predict([(query, c.chunk.content) for c in candidates])
        """
        import random
        for c in candidates:
            # Simulate rerank: slightly reshuffle similarity scores
            c.rerank_score = round(min(1.0, c.similarity_score + random.uniform(-0.05, 0.12)), 4)
        candidates.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return candidates[:5]  # top-5 after reranking

    async def _generate(self, query: str, chunks: list[RetrievedChunk]) -> dict:
        """
        GPT-4o generation with citation-aware prompt.

        Production code:
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=..., api_key=..., api_version="2024-02-01")

            sources_text = "\n\n".join([
                f"[{i+1}] {c.chunk.doc_title} — {c.chunk.section or 'General'}:\n{c.chunk.content}"
                for i, c in enumerate(chunks)
            ])
            prompt = RETRIEVAL_PROMPT_TEMPLATE.format(query=query, sources=sources_text)

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800,
            )
            result = json.loads(response.choices[0].message.content)
        """
        answer = self._simulate_answer(query, chunks)
        tokens = len(answer.split()) * 2
        cost = round((tokens / 1000) * 0.015 + (sum(len(c.chunk.content.split()) for c in chunks) / 1000) * 0.005, 5)

        # Determine confidence based on retrieval quality
        top_score = chunks[0].rerank_score if chunks else 0
        if top_score >= 0.65:
            confidence = "high"
            rationale = "Multiple highly relevant sources found with strong topical alignment."
        elif top_score >= 0.35:
            confidence = "medium"
            rationale = "Sources partially address the question; some details may require manual verification."
        else:
            confidence = "low"
            rationale = "Limited relevant sources retrieved. Manual review of primary guidelines recommended."

        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_rationale": rationale,
            "tokens": tokens,
            "cost": cost,
        }

    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations = []
        for i, rc in enumerate(chunks, 1):
            score = rc.rerank_score or rc.similarity_score
            confidence = "high" if score >= 0.6 else "medium" if score >= 0.35 else "low"
            citations.append(Citation(
                citation_id=i,
                doc_title=rc.chunk.doc_title,
                doc_type=rc.chunk.doc_type,
                section=rc.chunk.section,
                page_number=rc.chunk.page_number,
                excerpt=rc.chunk.content[:200] + "...",
                similarity_score=rc.similarity_score,
                confidence=confidence,
            ))
        return citations

    def _simulate_answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Simulate GPT-4o grounded answer. Replace with real API call in production."""
        q = query.lower()

        if any(w in q for w in ["flood", "water", "coastal"]):
            return (
                "Standard commercial property policies **exclude flood** as defined in ISO CP 10 30 [1]. "
                "Flood coverage may be offered as a manuscript endorsement, but eligibility is restricted: "
                "only properties in FEMA Flood Zone X or B qualify — Zones A, AE, V, and VE are non-eligible [1]. "
                "The maximum flood sublimit available is **$5M per occurrence** [1]. "
                "For coastal properties within 1,000 feet of tidal water, a separate **wind/hail deductible of 3–5% of TIV** applies [1]."
            )
        if any(w in q for w in ["tiv", "limit", "maximum", "large", "capacity", "reinsurance"]):
            return (
                "The maximum single-location TIV for habitational risks is **$50M** without reinsurance approval [1]. "
                "Properties exceeding this threshold must be submitted to **facultative reinsurance** with a minimum **45-day lead time** [1]. "
                "Business interruption coverage is capped at **12 months of gross profit** for standard accounts; "
                "extended BI up to 24 months may be offered to qualifying accounts with 3+ years of loss-free history [1]. "
                "For accounts requiring reinsurance, confirm the specific facultative treaty terms with the reinsurance desk before quoting."
            )
        if any(w in q for w in ["driver", "mvr", "auto", "fleet", "dui", "violation"]):
            return (
                "Driver risk is segmented into three tiers [2]. **Tier 1 (preferred)** requires a clear MVR for 5 years "
                "and fewer than 2 minor violations in 3 years [2]. **Tier 2 (standard)** allows 1 major violation or "
                "3 minor violations in 3 years [2]. **Tier 3 (non-standard)** requires senior approval and carries a "
                "**25% surcharge** on base premium [2]. Drivers with a DUI/DWI in the past **7 years are ineligible** [2]. "
                "The Q2 2024 book shows a 7.13% claim frequency with average severity of $18,400 [3]; "
                "accounts with 3+ at-fault claims in 36 months require senior referral at renewal [3]."
            )
        if any(w in q for w in ["cyber", "ransomware", "mfa", "backup", "endpoint"]):
            return (
                "All cyber applicants must complete the technology questionnaire (TQ-CY-2024) [4]. "
                "**Minimum qualifying controls** include: MFA on all email and remote access, EDR/XDR on 100% of endpoints, "
                "immutable offsite backups tested quarterly, and critical patches applied within 30 days [4]. "
                "**Accounts lacking MFA are ineligible** for cyber coverage [4]. "
                "Ransomware sublimits vary by sector: healthcare and financial services are capped at **$5M**, "
                "manufacturing and retail at **$10M** [5]. Accounts with a prior ransomware event within 24 months "
                "face a **5-year waiting period** before full sublimits apply [5]."
            )
        if any(w in q for w in ["sprinkler", "safeguard", "fire alarm", "protective"]):
            return (
                "Accounts with TIV above **$5M** must have a **central station monitored fire alarm system** [1]. "
                "Sprinkler systems are mandatory for restaurants/food service, warehouses exceeding 10,000 sq ft, "
                "and any occupancy with woodworking operations [1]. "
                "Failure to maintain required safeguards **voids the protective safeguard warranty** and may result "
                "in mid-term cancellation [1]. Confirm safeguard compliance during inspection before binding."
            )
        if any(w in q for w in ["workers comp", "workers' comp", "wc", "e-mod", "emod", "payroll", "classification"]):
            return (
                "The experience modification factor (e-mod) is calculated by NCCI or the applicable state bureau "
                "using **3 prior policy years** (excluding the most recent year) [2]. "
                "An e-mod above **1.25** requires senior underwriter approval and a mandatory risk improvement plan [2]. "
                "E-mods above **1.50** are ineligible for standard market — refer to specialty or E&S lines [2]. "
                "Key NCCI class codes: clerical (8810) at $0.42/$100 payroll, roofing contractors (5551) at $18.73/$100, "
                "restaurants (9082) at $3.18/$100 [2]. Annual payroll audits are required for all WC accounts [2]."
            )
        if any(w in q for w in ["liquor", "alcohol", "bar", "restaurant", "dram"]):
            return (
                "Liquor liability is **excluded from standard GL** for any insured whose operations involve "
                "manufacturing, selling, distributing, or serving alcoholic beverages [1]. "
                "A separate **liquor liability (dram shop) policy** is required for bars, restaurants, and event venues [1]. "
                "However, **host liquor liability** is included in standard GL for non-habitually-serving insureds "
                "(e.g., office holiday parties) up to **$1M per occurrence** [1]. "
                "Confirm whether the applicant's primary operations involve alcohol service to determine the correct coverage structure."
            )

        # Generic grounded answer
        relevant = chunks[:3] if chunks else []
        if relevant:
            excerpts = " ".join([f"Per {c.chunk.doc_title}, {c.chunk.content[:120]}... [{i+1}]" for i, c in enumerate(relevant)])
            return (
                f"Based on the retrieved underwriting documentation: {excerpts} "
                "Please verify specific limits and eligibility criteria with the current guideline edition before binding."
            )
        return (
            "The indexed documentation does not contain sufficient information to fully answer this question. "
            "Please consult the primary underwriting guidelines directly or escalate to senior underwriting."
        )
