"""
Document Ingestion Pipeline for Underwriting Copilot.

Handles the full document → chunk → embed → index flow.

In production this runs as an async Azure Function or Container App
triggered by Blob Storage events when new documents are uploaded.

Chunking strategy:
  - Semantic chunking (sentence-level boundaries, not fixed token windows)
  - Overlap: 10% of chunk size to preserve context at boundaries
  - Metadata extraction: section headers, page numbers, doc type
  - Chunk size: 512 tokens (optimal for text-embedding-3-large)

Embedding:
  - Azure OpenAI text-embedding-3-large (3072 dimensions)
  - Batch embedding (up to 2048 inputs per API call)
  - Stored in Azure AI Search with HNSW index

Production stack:
  - LlamaIndex for chunking + indexing orchestration
  - Azure AI Search as vector store (hybrid: BM25 + vector)
  - Azure OpenAI for embeddings
  - Azure Blob Storage for raw document storage
"""

import logging
import re
import uuid
from datetime import datetime
from typing import Iterator

from models.schemas import DocumentChunk, DocumentType, IndexingJob

logger = logging.getLogger(__name__)

# Chunking config
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 51  # ~10% overlap
AVG_CHARS_PER_TOKEN = 4  # rough approximation

CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * AVG_CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * AVG_CHARS_PER_TOKEN


class DocumentIngestionPipeline:
    """
    Ingests raw documents into the vector store.

    In production, replace _embed_chunks() with real Azure OpenAI calls
    and _index_chunks() with Azure AI Search upsert operations.

    Usage:
        pipeline = DocumentIngestionPipeline(vector_store, embedding_client)
        job = await pipeline.ingest(text_content, title, doc_type)
    """

    def __init__(self, vector_store=None, embedding_client=None):
        self.vector_store = vector_store      # Azure AI Search client
        self.embedding_client = embedding_client  # Azure OpenAI embeddings

    async def ingest(
        self,
        content: str,
        doc_title: str,
        doc_type: DocumentType,
        doc_id: Optional[str] = None,
    ) -> IndexingJob:
        """Full ingestion pipeline: chunk → embed → index."""
        doc_id = doc_id or str(uuid.uuid4())
        job = IndexingJob(
            doc_title=doc_title,
            doc_type=doc_type,
            status="chunking",
        )
        logger.info(f"[{job.job_id}] Starting ingestion: {doc_title}")

        try:
            # Stage 1: Chunk
            job.status = "chunking"
            chunks = list(self._chunk_document(content, doc_id, doc_title, doc_type))
            job.chunks_created = len(chunks)
            logger.info(f"[{job.job_id}] Created {len(chunks)} chunks")

            # Stage 2: Embed
            job.status = "embedding"
            embedded_chunks = await self._embed_chunks(chunks)

            # Stage 3: Index
            job.status = "indexing"
            await self._index_chunks(embedded_chunks)
            job.chunks_indexed = len(embedded_chunks)

            job.status = "complete"
            job.completed_at = datetime.utcnow()
            logger.info(f"[{job.job_id}] Ingestion complete: {len(chunks)} chunks indexed")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"[{job.job_id}] Ingestion failed: {e}", exc_info=True)

        return job

    def _chunk_document(
        self,
        content: str,
        doc_id: str,
        doc_title: str,
        doc_type: DocumentType,
    ) -> Iterator[DocumentChunk]:
        """
        Semantic chunking with section-header detection.

        In production, use LlamaIndex's SemanticSplitterNodeParser
        which uses embedding similarity to find natural chunk boundaries.

        Production code:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

            embed_model = AzureOpenAIEmbedding(model="text-embedding-3-large", ...)
            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model,
            )
            nodes = splitter.get_nodes_from_documents([document])
        """
        # Extract sections via header detection
        sections = self._extract_sections(content)

        chunk_index = 0
        for section_name, section_text in sections:
            # Split section into overlapping chunks
            start = 0
            while start < len(section_text):
                end = min(start + CHUNK_SIZE_CHARS, len(section_text))

                # Extend to sentence boundary if possible
                if end < len(section_text):
                    boundary = section_text.rfind('. ', start, end)
                    if boundary > start + CHUNK_SIZE_CHARS // 2:
                        end = boundary + 1

                chunk_text = section_text[start:end].strip()
                if len(chunk_text) < 50:  # Skip very small trailing chunks
                    break

                token_count = max(1, len(chunk_text) // AVG_CHARS_PER_TOKEN)

                yield DocumentChunk(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    section=section_name,
                    page_number=self._estimate_page(chunk_index),
                    content=chunk_text,
                    token_count=token_count,
                )
                chunk_index += 1
                start = end - CHUNK_OVERLAP_CHARS  # overlap
                if start >= end:
                    break

    def _extract_sections(self, content: str) -> list[tuple[str, str]]:
        """Extract named sections from document text."""
        # Match common insurance doc headers:
        # "2.1 Property Coverage Rules", "SECTION IV: EXCLUSIONS", etc.
        header_pattern = re.compile(
            r'^(?:(?:\d+\.)+\d*\s+|(?:SECTION|CHAPTER|PART)\s+[IVX\d]+[:\s]+)?'
            r'([A-Z][A-Z\s\-]{4,60})$',
            re.MULTILINE,
        )

        matches = list(header_pattern.finditer(content))
        if not matches:
            return [("General", content)]

        sections = []
        for i, match in enumerate(matches):
            section_name = match.group(0).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start:end].strip()
            if section_text:
                sections.append((section_name, section_text))

        return sections if sections else [("General", content)]

    def _estimate_page(self, chunk_index: int, chars_per_page: int = 3000) -> int:
        return (chunk_index * CHUNK_SIZE_CHARS // chars_per_page) + 1

    async def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[dict]:
        """
        Batch embed chunks using Azure OpenAI text-embedding-3-large.

        Production code:
            from openai import AzureOpenAI
            client = AzureOpenAI(...)
            texts = [c.content for c in chunks]
            # Batch in groups of 2048 (API limit)
            for batch in batched(texts, 2048):
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch,
                )
                embeddings = [e.embedding for e in response.data]
        """
        import random
        return [
            {
                "chunk": c,
                "embedding": [random.gauss(0, 0.1) for _ in range(3072)],  # simulated
            }
            for c in chunks
        ]

    async def _index_chunks(self, embedded_chunks: list[dict]) -> None:
        """
        Upsert chunks into Azure AI Search vector index.

        Production code (LlamaIndex):
            from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
            vector_store = AzureAISearchVectorStore(
                search_or_index_client=index_client,
                index_name="underwriting-docs",
                index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
                id_field_key="chunk_id",
                chunk_field_key="content",
                embedding_field_key="embedding",
                metadata_string_field_key="metadata",
                doc_id_field_key="doc_id",
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)

        Direct Azure AI Search SDK:
            search_client.upload_documents(documents=[
                {
                    "chunk_id": ec["chunk"].chunk_id,
                    "doc_id": ec["chunk"].doc_id,
                    "content": ec["chunk"].content,
                    "embedding": ec["embedding"],
                    "doc_title": ec["chunk"].doc_title,
                    "doc_type": ec["chunk"].doc_type,
                    "section": ec["chunk"].section,
                    "page_number": ec["chunk"].page_number,
                }
                for ec in embedded_chunks
            ])
        """
        logger.info(f"Indexed {len(embedded_chunks)} chunks into vector store")


# Optional: needed for _chunk_document type hint
from typing import Optional
