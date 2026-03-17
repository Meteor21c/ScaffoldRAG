import os
import gc
import json
import pdb
import pickle
import torch
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from config.config import (
    CACHE_DIR,
    RESULT_DIR,
    DENSE_MODEL_NAME,  # 新的稠密模型变量
    RERANK_MODEL_NAME,  # 新的重排序模型变量
    EMBEDDING_BATCH_SIZE,
    RETRIEVAL_TOP_K_CANDIDATES,  # 新增
    RERANK_TOP_K  # 新增
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class BaseRAG:
    def __init__(self, corpus_path: str = None, cache_dir: str = CACHE_DIR):
        """Initialize the BaseRAG system."""
        self.MODEL_NAME = "BaseRAG"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 1. 初始化稠密检索模型 (Dense Retriever)
        logger.info(f"Loading Dense Retriever: {DENSE_MODEL_NAME}...")
        self.model = SentenceTransformer(DENSE_MODEL_NAME)

        # 2. 初始化重排序模型 (Cross-Encoder Reranker)
        logger.info(f"Loading Reranker: {RERANK_MODEL_NAME}...")
        # CrossEncoder 会自动处理 query 和 document 的拼接
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)

        # 3. 初始化数据容器
        self.corpus = {}  # id -> text
        self.corpus_embeddings = None  # Tensor
        self.bm25 = None  # BM25Okapi 对象
        self.sentences = None  # List[str] 用于 BM25 和 Vanilla 检索
        self.retrieval_cache = {}

        # 默认 top_k，后续会被 config 覆盖或动态调整
        self.top_k = 5

        if corpus_path:
            self.load_corpus(corpus_path)

    def load_corpus(self, corpus_path: str):
        """Load and process the document corpus."""
        logger.info("Loading corpus...")
        with open(corpus_path, 'r') as f:
            documents = json.load(f)

        # Process documents into chunks
        self.corpus = {
            i: f"Title: {doc['title']}. Content: {doc['text']}"
            for i, doc in enumerate(documents)
        }

        # Store sentences for vanilla retrieval
        self.sentences = list(self.corpus.values())

        # ---------------------------------------------------------
        # A. 构建/加载 稠密向量索引 (Dense Index)
        # ---------------------------------------------------------
        dense_cache_file = os.path.join(self.cache_dir,
                                        f'embeddings_{len(self.corpus)}_{DENSE_MODEL_NAME.replace("/", "_")}.pt')

        if os.path.exists(dense_cache_file):
            logger.info("Loading cached Dense embeddings...")
            self.corpus_embeddings = torch.load(dense_cache_file)
        else:
            logger.info("Computing Dense embeddings (this may take a while)...")
            self.corpus_embeddings = self.encode_sentences_batch(self.sentences)
            torch.save(self.corpus_embeddings, dense_cache_file)
            logger.info("Dense embeddings saved.")

        # ---------------------------------------------------------
        # B. 构建/加载 稀疏索引 (Sparse Index - BM25)
        # ---------------------------------------------------------
        bm25_cache_file = os.path.join(self.cache_dir, f'bm25_{len(self.corpus)}.pkl')

        if os.path.exists(bm25_cache_file):
            logger.info("Loading cached BM25 index...")
            with open(bm25_cache_file, 'rb') as f:
                self.bm25 = pickle.load(f)
        else:
            logger.info("Building BM25 index...")
            # 简单的分词策略：转小写并按空格分割 (针对英文数据集足够，中文需要 jieba)
            tokenized_corpus = [doc.lower().split() for doc in tqdm(self.sentences, desc="Tokenizing for BM25")]
            self.bm25 = BM25Okapi(tokenized_corpus)

            with open(bm25_cache_file, 'wb') as f:
                pickle.dump(self.bm25, f)
            logger.info("BM25 index saved.")

        # 兼容性字段 (For compatibility)
        self.embeddings = self.corpus_embeddings

    def encode_sentences_batch(self, sentences: List[str]) -> torch.Tensor:
        """Encode sentences in batches with memory management."""
        # 使用 config 中定义的 batch_size
        batch_size = EMBEDDING_BATCH_SIZE
        all_embeddings = []

        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
            batch = sentences[i:i + batch_size]

            # 主动垃圾回收，防止显存碎片化
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True #20250202 ADD
                )
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings

        # 保留原有的辅助方法，暂时不做大改动，Retrieve 方法将在 Step 3 中重写

    def set_top_k(self, top_k: int):
        self.top_k = top_k

    def encode_batch(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Encode texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings)

    def encode_sentences_batch(self, sentences: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode sentences in batches with memory management."""
        all_embeddings = []

        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding sentences"):
            batch = sentences[i:i + batch_size]

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)

        final_embeddings = torch.cat(all_embeddings, dim=0)
        del all_embeddings
        gc.collect()

        return final_embeddings

    def build_index(self, sentences: List[str], batch_size: int = 32):
        """Build the embedding index for the sentences."""
        self.sentences = sentences

        # Try to load existing embeddings
        embedding_file = f'cache/embeddings_{len(sentences)}.pkl'
        if os.path.exists(embedding_file):
            try:
                with open(embedding_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Embeddings loaded from {embedding_file}")
                return
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")

        # Build new embeddings
        self.embeddings = self.encode_sentences_batch(sentences, batch_size)

        # Save embeddings
        try:
            os.makedirs('cache', exist_ok=True)
            with open(embedding_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def _search_dense(self, query: str, top_k: int) -> List[int]:
        """第一路召回：Dense Retrieval (Vector Search)"""
        if self.corpus_embeddings is None:
            return []

        with torch.no_grad():
            # 1. 编码 Query
            query_embedding = self.model.encode([query], convert_to_tensor=True)
            if torch.cuda.is_available():
                query_embedding = query_embedding.to('cpu')  # 确保在 CPU 上计算相似度以节省显存

            # 2. 计算 Cosine Similarity
            # self.corpus_embeddings 应该已经在 CPU 上 (由 Step 2 保证)
            cos_scores = torch.nn.functional.cosine_similarity(
                query_embedding,
                self.corpus_embeddings
            )

            # 3. 获取 Top K 索引
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.sentences)))
            return top_results.indices.tolist()

    def _search_sparse(self, query: str, top_k: int) -> List[int]:
        """第二路召回：Sparse Retrieval (BM25)"""
        if self.bm25 is None:
            return []

        # 1. 对 Query 进行分词 (必须与索引构建时的分词逻辑一致)
        tokenized_query = query.lower().split()

        # 2. 获取所有文档的得分
        scores = self.bm25.get_scores(tokenized_query)

        # 3. 获取 Top K 索引 (使用 argsort 获取最大得分的索引)
        # argsort 是升序，所以取最后 k 个并反转
        top_n_indices = np.argsort(scores)[-top_k:][::-1]
        return top_n_indices.tolist()

    def _rrf_fusion(self, dense_indices: List[int], sparse_indices: List[int],
                    weights: Dict[str, float] = {'dense': 1.0, 'sparse': 0},  # <--- 新增权重参数
                    k: int = 60) -> List[int]:
        """
        加权倒数排名融合 (Weighted Reciprocal Rank Fusion)
        """
        rrf_score = {}

        # 处理 Dense 结果 (给予更高权重)
        for rank, idx in enumerate(dense_indices):
            if idx not in rrf_score:
                rrf_score[idx] = 0
            # weight * (1 / (k + rank))
            rrf_score[idx] += weights['dense'] * (1 / (k + rank + 1))

        # 处理 Sparse 结果
        for rank, idx in enumerate(sparse_indices):
            if idx not in rrf_score:
                rrf_score[idx] = 0
            rrf_score[idx] += weights['sparse'] * (1 / (k + rank + 1))

        # 按分数降序排序
        sorted_indices = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
        return sorted_indices

        # 按分数降序排序
        sorted_indices = sorted(rrf_score.keys(), key=lambda x: rrf_score[x], reverse=True)
        return sorted_indices

    def _rerank(self, query: str, candidate_indices: List[int], top_k: int) -> List[str]:
        """使用 Cross-Encoder 进行重排序"""
        if not candidate_indices:
            return []

        # 1. 准备 Pairs: [[query, doc1], [query, doc2], ...]
        candidate_docs = [self.sentences[idx] for idx in candidate_indices]
        pairs = [[query, doc] for doc in candidate_docs]

        # 2. Cross-Encoder 推理打分
        # 注意：Cross-Encoder 通常在 GPU 上运行会快很多
        with torch.no_grad():
            scores = self.reranker.predict(pairs, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)

        # 3. 排序并获取最终 Top K 文档文本
        # 将 (index, score) 结合
        doc_scores = list(zip(candidate_indices, scores))

        # 按分数降序排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回最终的文本内容
        final_top_k_indices = [idx for idx, score in doc_scores[:top_k]]
        final_results = [self.sentences[idx] for idx in final_top_k_indices]

        return final_results

    def retrieve(self, query: str) -> List[str]:
        """
                [Advanced] Hybrid Retrieval + Rerank Pipeline with Smart Caching
                """
        # 0. 智能缓存检查 (防止缓存陷阱)
        if query in self.retrieval_cache:
            cached_results = self.retrieval_cache[query]
            # 只有当缓存的数据量足够满足当前需求时，才使用缓存
            if len(cached_results) >= self.top_k:
                return cached_results[:self.top_k]
            # 否则：缓存不够用（比如之前存了5个，现在要10个），需要重新计算

        # 1. 双路召回 (Parallel Recall)
        dense_hits = self._search_dense(query, top_k=RETRIEVAL_TOP_K_CANDIDATES)
        sparse_hits = self._search_sparse(query, top_k=RETRIEVAL_TOP_K_CANDIDATES)

        # 2. 融合 (Fusion)
        fused_indices = self._rrf_fusion(dense_hits, sparse_hits,weights={'dense': 1.0, 'sparse': 0})
        rerank_candidates = fused_indices[:RETRIEVAL_TOP_K_CANDIDATES]

        # 3. 重排序 (Rerank)
        # 策略：取 RERANK_TOP_K 和 self.top_k 的最大值，确保缓存里有足够的数据做"蓄水池"
        # 这样即使后续步骤需要 expand search window，也不用频繁重新 rerank
        actual_rerank_k = max(RERANK_TOP_K, self.top_k)
        reranked_results = self._rerank(query, rerank_candidates, top_k=actual_rerank_k)
        # 4. 缓存并返回
        self.retrieval_cache[query] = reranked_results

        # 严格遵守本次调用的 top_k 限制
        return reranked_results[:self.top_k]

    # def retrieve(self, query: str) -> List[str]:
    #     """
    #     [Advanced] Hybrid Retrieval + Rerank Pipeline
    #     """
    #     if query in self.retrieval_cache:
    #         return self.retrieval_cache[query]
    #
    #     # 1. 双路召回
    #     dense_hits = self._search_dense(query, top_k=RETRIEVAL_TOP_K_CANDIDATES)
    #     sparse_hits = self._search_sparse(query, top_k=RETRIEVAL_TOP_K_CANDIDATES)
    #
    #     # 2. 融合 (Fusion) - 引入权重调整
    #     # 这里我们更信任 BGE-M3，因此给 dense 0.7, sparse 0.3
    #     fused_indices = self._rrf_fusion(
    #         dense_hits,
    #         sparse_hits,
    #         weights={'dense': 0.8, 'sparse': 0.2}  # <--- 调整权重
    #     )
    #
    #     rerank_candidates = fused_indices[:RETRIEVAL_TOP_K_CANDIDATES]
    #
    #     # 3. 重排序
    #     final_results = self._rerank(query, rerank_candidates, top_k=RERANK_TOP_K)
    #
    #     self.retrieval_cache[query] = final_results
    #     return final_results

    # def retrieve(self, query: str) -> List[str]:
    #     """Retrieve similar sentences using query embedding."""
    #     # Check cache first
    #     if query in self.retrieval_cache:
    #         return self.retrieval_cache[query]
    #     # pdb.set_trace()
    #
    #     if self.corpus_embeddings is None or not self.corpus:
    #         return []
    #
    #     try:
    #         # Encode query
    #         with torch.no_grad():
    #             query_embedding = self.model.encode([query], convert_to_tensor=True)[0]
    #             query_embedding = query_embedding.cpu()
    #
    #         # Calculate similarities
    #         similarities = torch.nn.functional.cosine_similarity(
    #             query_embedding.unsqueeze(0),
    #             self.corpus_embeddings
    #         )
    #
    #         # Convert indices to list before using them
    #         top_k_scores, top_k_indices = similarities.topk(self.top_k)
    #         indices = top_k_indices.tolist()
    #
    #         # Get results using integer indices
    #         results = [self.corpus[idx] for idx in indices]
    #
    #         # Cache results
    #         self.retrieval_cache[query] = results
    #         return results
    #
    #     except Exception as e:
    #         logger.error(f"Error in retrieve: {e}")
    #         return []