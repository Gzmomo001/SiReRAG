from typing import Any, Dict, List, Optional

import asyncio
from enum import Enum
from tqdm import tqdm

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever, QueryType
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.ingestion import run_transformations
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    BasePydanticVectorStore,
)
from llama_index.packs.raptor.clustering import get_clusters

from llama_index.llms.deepseek import DeepSeek

# llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DS_API"))


from llama_index.embeddings.ollama import OllamaEmbedding

# ollama_embedding = OllamaEmbedding(
#     model_name="nomic-embed-text:latest",
#     base_url="http://localhost:11434",
# )


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text, including as many key details as needed."
)


class QueryModes(str, Enum):
    """Query modes."""

    tree_traversal = "tree_traversal"
    collapsed = "collapsed"


class SummaryModule(BaseModel):
    response_synthesizer: BaseSynthesizer = Field(description="LLM")
    summary_prompt: str = Field(
        default=DEFAULT_SUMMARY_PROMPT,
        description="Summary prompt.",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to generate summaries."
    )
    show_progress: bool = Field(default=True, description="Show progress.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        num_workers: int = 32,  # 增加默认并发数
    ) -> None:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", 
            use_async=True, 
            llm=llm,
            streaming=False,  # 关闭流式处理以提高速度
        )
        super().__init__(
            response_synthesizer=response_synthesizer,
            summary_prompt=summary_prompt,
            num_workers=num_workers,
        )

    async def generate_summaries(
        self, documents_per_cluster: List[List[BaseNode]]
    ) -> List[str]:
        """Generate summaries of documents per cluster."""
        print(f"\n=== 开始生成摘要 ===")
        print(f"总集群数: {len(documents_per_cluster)}")
        print(f"工作线程数: {self.num_workers}")
        
        # 创建所有任务
        tasks = []
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            tasks.append(self.response_synthesizer.asynthesize(self.summary_prompt, with_scores))

        # 使用更大的批处理
        responses = []
        completed = 0
        batch_size = self.num_workers * 4  # 增加批处理大小
        
        print("生成摘要")
        semaphore = asyncio.Semaphore(self.num_workers * 2)  # 控制并发数
        
        async def process_batch(batch):
            async with semaphore:
                return await asyncio.gather(*batch, return_exceptions=True)
        
        # 创建所有批次的任务
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        batch_tasks = [process_batch(batch) for batch in batches]
        
        # 并发执行所有批次
        with tqdm(total=len(tasks)) as pbar:
            for batch_responses in await asyncio.gather(*batch_tasks):
                for response in batch_responses:
                    if isinstance(response, Exception):
                        print(f"\n摘要生成失败: {str(response)}")
                        responses.append("摘要生成失败")
                    else:
                        responses.append(response)
                        print(f"\n摘要生成成功:")
                    pbar.update(1)
                    completed += 1

        print(f"\n=== 摘要生成完成 ===")
        print(f"成功处理: {len([r for r in responses if r != '摘要生成失败'])}/{len(tasks)}")
        
        return [str(response) for response in responses]


class RaptorRetriever(BaseRetriever):
    """Raptor indexing retriever."""

    def __init__(
        self,
        documents: List[BaseNode],
        higher_level_facts: List[BaseNode],
        tree_depth: int = 3,
        similarity_top_k: int = 2,
        llm: Optional[DeepSeek] = None,
        embed_model: Optional[OllamaEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        transformations: Optional[List[TransformComponent]] = None,
        summary_module: Optional[SummaryModule] = None,
        existing_index: Optional[VectorStoreIndex] = None,
        mode: QueryModes = "collapsed",
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super().__init__(
            **kwargs,
        )
        
        print(f"Initializing RaptorRetriever with:")
        print(f"- Number of input documents: {len(documents)}")
        print(f"- Number of higher level facts: {len(higher_level_facts)}")
        print(f"- Tree depth: {tree_depth}")
        print(f"- Similarity top k: {similarity_top_k}")
        print(f"- Mode: {mode}")
        print(f"- Using LLM: {llm.__class__.__name__ if llm else 'None'}")
        print(f"- Using Embedding model: {embed_model.__class__.__name__ if embed_model else 'None'}")

        self.mode = mode
        self.summary_module = summary_module or SummaryModule(llm=llm)
        self.index = existing_index or VectorStoreIndex(
            nodes=[],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
            transformations=transformations,
        )
        self.tree_depth = tree_depth
        self.similarity_top_k = similarity_top_k
        self.rst_tree = {}
        self.higher_level_facts = higher_level_facts
        self.embedding_cache = {}  # 添加嵌入缓存字典
        if len(documents) > 0:
            asyncio.run(self.insert(documents))

    async def get_embeddings_batch(self, nodes: List[BaseNode], batch_size: int = 50) -> List[List[float]]:
        """带缓存的批量嵌入处理"""
        all_embeddings = []
        batch_to_process = []
        batch_indices = []
        
        # 检查缓存
        for i, node in enumerate(nodes):
            content = node.get_content(metadata_mode="embed")
            if content in self.embedding_cache:
                all_embeddings.append(self.embedding_cache[content])
            else:
                batch_to_process.append(content)
                batch_indices.append(i)
                
            # 当积累足够的未缓存项时，进行批处理
            if len(batch_to_process) >= batch_size:
                embeddings = await self._process_batch(batch_to_process)
                # 更新缓存和结果
                for content, embedding in zip(batch_to_process, embeddings):
                    self.embedding_cache[content] = embedding
                batch_to_process = []
                batch_indices = []
        
        # 处理剩余的项
        if batch_to_process:
            embeddings = await self._process_batch(batch_to_process)
            # 更新缓存和结果
            for content, embedding in zip(batch_to_process, embeddings):
                self.embedding_cache[content] = embedding
        
        return all_embeddings

    async def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """处理单个批次的嵌入"""
        embed_model = self.index._embed_model
        try:
            return await embed_model.aget_text_embedding_batch(batch)
        except Exception as e:
            print(f"嵌入处理失败: {str(e)}")
            try:
                print("重试中...")
                return await embed_model.aget_text_embedding_batch(batch)
            except Exception as e:
                print(f"重试失败: {str(e)}")
                raise e

    async def insert(self, documents: List[BaseNode]) -> None:
        """Given a set of documents, this function inserts higher level of abstractions within the index.

        For later retrieval

        Args:
            documents (List[BaseNode]): List of Documents
        """
        print("Fact+Raptor+left_right_only_fact_aggregate!!!")
        embed_model = self.index._embed_model
        transformations = self.index._transformations


        print("\n=== Starting Document Processing ===")
        print(f"Processing {len(documents)} documents")
        print(f"Processing {len(self.higher_level_facts)} higher level facts")
        
        embed_model = self.index._embed_model
        transformations = self.index._transformations

        print("\n=== Running Transformations ===")
        higher_facts = run_transformations(self.higher_level_facts, transformations, in_place=False)
        cur_nodes = run_transformations(documents, transformations, in_place=False)
        print(f"Transformed higher facts: {len(higher_facts)}")
        print(f"Transformed documents: {len(cur_nodes)}")
        
        self.rst_tree[0] = [c.text for c in cur_nodes]

        # 添加批处理和错误处理逻辑
        async def get_embeddings_batch(nodes: List[BaseNode], batch_size: int = 50) -> List[List[float]]:
            all_embeddings = []
            print(f"Processing embedding batch")
            for i in tqdm(range(0, len(nodes), batch_size)):
                batch = nodes[i:i + batch_size]
                try:
                    # print(f"Processing embedding batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
                    batch_embeddings = await embed_model.aget_text_embedding_batch(
                        [node.get_content(metadata_mode="embed") for node in batch]
                    )
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                    # 如果发生错误，重试该批次
                    try:
                        print("Retrying batch...")
                        batch_embeddings = await embed_model.aget_text_embedding_batch(
                            [node.get_content(metadata_mode="embed") for node in batch]
                        )
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        print(f"Retry failed: {str(e)}")
                        raise e
            return all_embeddings

        # Fact Tree
        print("\n=== Building Fact Tree ===")
        for level in range(self.tree_depth):
            print(f"\nProcessing Level {level}:")
            print(f"- Number of facts to process: {len(higher_facts)}")
            
            if self._verbose:
                print(f"Generating embeddings for level {level}.")

            # 使用新的批处理函数
            embeddings = await get_embeddings_batch(higher_facts)
            
            assert len(embeddings) == len(higher_facts)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(higher_facts, embeddings)
            }

            if self._verbose:
                print(f"Performing clustering for level {level}.")

            # cluster the documents
            nodes_per_cluster = get_clusters(higher_facts, id_to_embedding)

            if self._verbose:
                print(
                    f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
                )
            summaries_per_cluster = await self.summary_module.generate_summaries(
                nodes_per_cluster
            )

            if self._verbose:
                print(
                    f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}"
                )

            # replace the current nodes with their summaries
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]

            # insert the nodes with their embeddings and parent_id
            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            self.index.insert_nodes(nodes_with_embeddings)
            
            # set the current nodes to the new nodes
            higher_facts = new_nodes
            self.rst_tree[level+1] = [c.text for c in higher_facts]
        self.index.insert_nodes(higher_facts)

        # Raptor Tree
        print("\n=== Building Raptor Tree ===")
        for level in range(self.tree_depth):
            print(f"\nProcessing Level {level}:")
            print(f"- Number of nodes to process: {len(cur_nodes)}")
            
            if self._verbose:
                print(f"Generating embeddings for level {level}.")

            # 使用新的批处理函数
            embeddings = await get_embeddings_batch(cur_nodes)
            
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {
                node.id_: embedding for node, embedding in zip(cur_nodes, embeddings)
            }

            if self._verbose:
                print(f"Performing clustering for level {level}.")

            # cluster the documents
            nodes_per_cluster = get_clusters(cur_nodes, id_to_embedding)

            if self._verbose:
                print(
                    f"Generating summaries for level {level} with {len(nodes_per_cluster)} clusters."
                )
            summaries_per_cluster = await self.summary_module.generate_summaries(
                nodes_per_cluster
            )

            if self._verbose:
                print(
                    f"Level {level} created summaries/clusters: {len(nodes_per_cluster)}"
                )

            # replace the current nodes with their summaries
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]

            # insert the nodes with their embeddings and parent_id
            nodes_with_embeddings = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            self.index.insert_nodes(nodes_with_embeddings)

            # set the current nodes to the new nodes
            cur_nodes = new_nodes
            self.rst_tree[level+1] = [c.text for c in cur_nodes]

        self.index.insert_nodes(cur_nodes)

    async def collapsed_retrieval(self, query_str: str) -> Response:
        """Query the index as a collapsed tree."""
        print(f"\n=== Collapsed Retrieval ===")
        print(f"Query: {query_str}")
        response = await self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        ).aretrieve(query_str)
        print(f"Retrieved {len(response)} nodes")
        return response

    async def tree_traversal_retrieval(self, query_str: str) -> Response:
        """Query the index as a tree."""
        print(f"\n=== Tree Traversal Retrieval ===")
        print(f"Query: {query_str}")
        print(f"Starting from level: {self.tree_depth - 1}")
        
        # get top k nodes for each level, starting with the top
        parent_ids = None
        nodes = []
        level = self.tree_depth - 1
        while level >= 0:
            # retrieve nodes at the current level
            if parent_ids is None:
                nodes = await self.index.as_retriever(
                    similarity_top_k=self.similarity_top_k,
                    filters=MetadataFilters(
                        filters=[MetadataFilter(key="level", value=level)]
                    ),
                ).aretrieve(query_str)

                parent_ids = [node.id_ for node in nodes]
                if self._verbose:
                    print(f"Retrieved parent IDs from level {level}: {parent_ids!s}")
            # retrieve nodes that are children of the nodes at the previous level
            elif parent_ids is not None and len(parent_ids) > 0:
                nested_nodes = await asyncio.gather(
                    *[
                        self.index.as_retriever(
                            similarity_top_k=self.similarity_top_k,
                            filters=MetadataFilters(
                                filters=[MetadataFilter(key="parent_id", value=id_)]
                            ),
                        ).aretrieve(query_str)
                        for id_ in parent_ids
                    ]
                )

                nodes = [node for nested in nested_nodes for node in nested]

                if self._verbose:
                    print(f"Retrieved {len(nodes)} from parents at level {level}.")

                level -= 1
                parent_ids = None

        return nodes

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        # not used, needed for type checking

    def retrieve(
        self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle

        return asyncio.run(self.aretrieve(query_str, mode or self.mode))

    async def aretrieve(
        self, query_str_or_bundle: QueryType, mode: Optional[QueryModes] = None
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        if isinstance(query_str_or_bundle, QueryBundle):
            query_str = query_str_or_bundle.query_str
        else:
            query_str = query_str_or_bundle

        mode = mode or self.mode
        if mode == "tree_traversal":
            return await self.tree_traversal_retrieval(query_str)
        elif mode == "collapsed":
            return await self.collapsed_retrieval(query_str)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def persist(self, persist_dir: str) -> None:
        self.index.storage_context.persist(persist_dir=persist_dir)

    @classmethod
    def from_persist_dir(
        cls: "RaptorRetriever",
        persist_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        **kwargs: Any,
    ) -> "RaptorRetriever":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return cls(
            [],
            existing_index=load_index_from_storage(
                storage_context, embed_model=embed_model
            ),
            **kwargs,
        )


class RaptorPack(BaseLlamaPack):
    """Raptor pack."""

    def __init__(
        self,
        documents: List[BaseNode],
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        similarity_top_k: int = 2,
        mode: QueryModes = "collapsed",
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        self.retriever = RaptorRetriever(
            documents,
            embed_model=embed_model,
            llm=llm,
            similarity_top_k=similarity_top_k,
            vector_store=vector_store,
            mode=mode,
            verbose=verbose,
            **kwargs,
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {
            "retriever": self.retriever,
        }

    def run(
        self,
        query: str,
        mode: Optional[QueryModes] = None,
    ) -> Any:
        """Run the pipeline."""
        return self.retriever.retrieve(query, mode=mode)
