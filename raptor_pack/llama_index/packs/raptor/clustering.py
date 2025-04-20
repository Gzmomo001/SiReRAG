"""
Minorly tweaked from https://github.com/parthsarthi03/raptor/blob/master/raptor/cluster_tree_builder.py.

Full credits to the original authors!
"""

import numpy as np
import random
import tiktoken
import umap
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

from llama_index.core.schema import BaseNode


# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


# 检查并设置 MPS 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    # 转换为 PyTorch tensor 并移动到 MPS
    embeddings_tensor = torch.tensor(embeddings, device=device)
    
    # 数据预处理
    mean = embeddings_tensor.mean(dim=0, keepdim=True)
    std = embeddings_tensor.std(dim=0, keepdim=True)
    embeddings_tensor = (embeddings_tensor - mean) / (std + 1e-8)
    
    if n_neighbors is None:
        n_neighbors = min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)
    
    try:
        # 转回 numpy 进行 UMAP 处理
        embeddings_np = embeddings_tensor.cpu().numpy()
        reduced = umap.UMAP(
            n_neighbors=n_neighbors, 
            n_components=dim, 
            metric=metric,
            random_state=RANDOM_SEED
        ).fit_transform(embeddings_np)
        return reduced
    except Exception as e:
        print(f"UMAP降维失败: {str(e)}")
        # 如果UMAP失败，返回简单的降维结果
        return embeddings_tensor[:, :dim].cpu().numpy() if embeddings_tensor.shape[1] > dim else embeddings_tensor.cpu().numpy()


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    # 转换为 PyTorch tensor
    embeddings_tensor = torch.tensor(embeddings, device=device)
    
    # 处理无效值
    embeddings_tensor = torch.nan_to_num(embeddings_tensor)
    
    try:
        # 转回 numpy 进行 GMM 聚类
        embeddings_np = embeddings_tensor.cpu().numpy()
        n_clusters = get_optimal_clusters(embeddings_np)
        gm = GaussianMixture(
            n_components=n_clusters, 
            random_state=random_state,
            reg_covar=1e-3,
            max_iter=200,
            n_init=5
        )
        gm.fit(embeddings_np)
        probs = gm.predict_proba(embeddings_np)
        
        # 概率计算移到 GPU
        probs_tensor = torch.tensor(probs, device=device)
        labels = [torch.where(prob > threshold)[0].cpu().numpy() for prob in probs_tensor]
        return labels, n_clusters
    except Exception as e:
        print(f"GMM聚类失败: {str(e)}")
        return [np.array([0]) for _ in range(len(embeddings))], 1


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # 转换为 PyTorch tensor
    embeddings_tensor = torch.tensor(embeddings, device=device)
    
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        # 使用 PyTorch 进行布尔索引
        mask = torch.tensor([i in gc for gc in global_clusters], device=device)
        global_cluster_embeddings_ = embeddings_tensor[mask].cpu().numpy()

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            # 使用 PyTorch 进行相等性比较
            embeddings_tensor_comp = torch.tensor(local_cluster_embeddings_, device=device)
            indices = torch.where(
                (embeddings_tensor.unsqueeze(1) == embeddings_tensor_comp).all(-1)
            )[0].cpu().numpy()
            
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def get_clusters(
    nodes: List[BaseNode],
    embedding_map: Dict[str, List[List[float]]],
    max_length_in_cluster: int = 10000,
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base"),
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,
) -> List[List[BaseNode]]:
    if len(nodes) <= 1:
        return [nodes]
        
    try:
        embeddings = np.array([np.array(embedding_map[node.id_]) for node in nodes])
        
        # 数据预处理
        embeddings = np.nan_to_num(embeddings)
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        
        # 执行聚类
        clusters = perform_clustering(
            embeddings, 
            dim=min(reduction_dimension, embeddings.shape[1]), 
            threshold=threshold
        )
        
        # Initialize an empty list to store the clusters of nodes
        node_clusters = []
    except Exception as e:
        print(f"聚类过程发生错误: {str(e)}")
        return [nodes]

    # Iterate over each unique label in the clusters
    for label in np.unique(np.concatenate(clusters)):
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]

        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to recluster it
        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, recluster this cluster
        # If the total length did not change from the previous call then don't try again to avoid infinite recursion!
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            node_clusters.extend(
                get_clusters(
                    cluster_nodes,
                    embedding_map,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    prev_total_length=total_length,
                )
            )
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters
