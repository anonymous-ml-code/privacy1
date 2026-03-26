import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.linalg import eigh, svd
from torch.linalg import svdvals
from EfficientKDE import EfficientKDE
def enhanced_manifold_mcmc_sampler(
        self, X: torch.Tensor, bw: float = 0.1, num_samples: int = 100,
        k_neighbors: int = 20, epsilon: float = 1e-3, sigma: float = 0.05,
        burn_in: int = 100, num_chains: int = 3, verbose: bool = False, # NEW: Exploration parameters
        exploration_decay: float = 0.99,    # Gradually reduce exploration
        mode_hopping_prob: float = 0.05,    # Probability to jump to new mode
        adaptive_target_rate: float = 0.234, # Optimal acceptance rate
) -> torch.Tensor:
    device = X.device
    N, D = X.shape
    kde = EfficientKDE(X, bw_method="dimension_scaled")

    # Initialize multiple chains
    with torch.no_grad():
        # initial_log_pdf = self.mvn.log_prob(X)  # Pass X and bw
        initial_log_pdf = kde.log_pdf(X)
        _, init_indices = torch.topk(initial_log_pdf, num_chains)
        initial_points = X[init_indices]

    samples = []
    acceptance_rates = []
    manifold_dim = estimate_manifold_dimension_pytorch(X)
    # Ensure manifold dimension is reasonable
    manifold_dim = min(manifold_dim, D - 1)  # Must be less than D for exploration
    manifold_dim = max(manifold_dim, 1)  # At least 1
    exploration_factor = 0.1
    # Compute log-PDF with different normalization strategies
    # log_pdf_standard = kde.log_pdf(query_points, normalization="standard")
    # log_pdf_adaptive = kde.log_pdf(query_points, normalization="adaptive")
    # log_pdf_dimfree = kde.log_pdf(query_points, normalization="dimension_free")
    #
    # print(f"Bandwidth: {kde.bw:.4f}")
    # print(f"Effective dimension: {estimate_effective_dimension(X):.1f}")
    for chain in range(num_chains):
        current_x = initial_points[chain].clone()
        chain_acceptances = []
        current_acceptance_rate = None

        for iter in range(burn_in + num_samples // num_chains):
            # Calculate current acceptance rate (window of last 50 iterations)
            # NEW: Adaptive exploration scheduling
            current_exploration = exploration_factor * (exploration_decay ** iter)


            if len(chain_acceptances) >= 50:
                current_acceptance_rate = torch.tensor(chain_acceptances[-50:]).float().mean().item()

            # Get adaptive parameters with explicit base values
            current_bw, current_epsilon = adaptive_sampling_parameters(
                X=X,
                current_x=current_x,
                iteration=iter,
                max_iterations=burn_in + num_samples // num_chains,
                base_bw=bw,  # Pass the initial bw
                base_epsilon=epsilon,  # Pass the initial epsilon
                acceptance_rate=current_acceptance_rate
            )

            # Estimate tangent space
            tangent_basis = estimate_tangent_space_improved(
                current_x, X, k_neighbors, manifold_dim
            )

            # Enhanced proposal
            delta = manifold_aware_proposal_robust(
                kde, current_x, X, tangent_basis, epsilon, sigma, exploration_factor=current_exploration
            )

            # NEW: Occasional mode hopping to escape local minima
            if torch.rand(1).item() < mode_hopping_prob:
                candidate_x = _mode_hop(kde, X, samples, current_x)
            else:
                candidate_x = current_x + delta

            temperature = max(0.3, 1.0 - iter / (burn_in + num_samples // num_chains) * 0.7)
            log_p_current = kde.log_pdf(current_x)/ temperature
            log_p_candidate = kde.log_pdf(candidate_x)/ temperature
            accept_prob = torch.min(torch.tensor(1.0),
                                    torch.exp(log_p_candidate - log_p_current))

            if torch.rand(1, device=device) < accept_prob:
                current_x = candidate_x
                chain_acceptances.append(1.0)
            else:
                chain_acceptances.append(0.0)

            # Collect samples
            if iter >= burn_in:
                samples.append(current_x.clone())

            if verbose and iter % 100 == 0:
                current_rate = torch.tensor(chain_acceptances[-100:]).mean() if len(
                    chain_acceptances) >= 100 else torch.tensor(chain_acceptances).mean()
                print(f"Chain {chain}, Iter {iter}, Accept Rate: {current_rate:.3f}, "
                      f"BW: {current_bw:.4f}, Epsilon: {current_epsilon:.6f}")

        chain_final_rate = torch.tensor(chain_acceptances).mean()
        acceptance_rates.append(chain_final_rate)
        if verbose:
            print(f"Chain {chain} final acceptance rate: {chain_final_rate:.3f}")

    final_samples = torch.stack(samples)

    # Optional: Remove duplicates and add diversity if needed
    if len(final_samples) > num_samples:
        final_samples = select_diverse_samples(final_samples, num_samples)

    return final_samples, manifold_dim

def estimate_manifold_dimension_pytorch(
        X: torch.Tensor,
        k: int = 20,  # 近邻数（建议 \(k = 5d \sim 10d\)）
        M: int = 1000,  # 采样样本数（减少计算量）
        device: torch.device = None
) -> int:
    """
    基于PyTorch张量的流形维度估计（无外部依赖）
    Args:
        X: 输入样本集，形状 (N, D)，PyTorch张量（支持CPU/GPU）
        k: 近邻数（用于局部PCA）
        M: 随机采样的样本数（用于估计局部维度）
        device: 计算设备（默认与X相同）
    Returns:
        估计的流形维度 d
    """
    if device is None:
        device = X.device
    X = X.to(device)  # 确保输入张量在目标设备上
    N, D = X.shape
    M = min(M, N)  # 若样本数不足M，使用全部样本

    # -------------------------- 1. 随机采样样本 -------------------------
    sample_indices = torch.randperm(N, device=device)[:M]  # 随机采样M个样本索引
    X_sample = X[sample_indices]  # (M, D)

    # -------------------------- 2. 近邻搜索（PyTorch实现） -------------------------
    # 计算采样样本与所有样本的欧氏距离矩阵 (M, N)
    dist_matrix = torch.cdist(X_sample, X, p=2)  # 欧氏距离（L2）

    # 取每个采样样本的k近邻索引（排除自身）
    # 若存在重复样本（距离为0），确保至少取k个不同近邻
    _, knn_indices = torch.topk(dist_matrix, k=k + 1, largest=False)  # 取前k+1个（含自身）
    knn_indices = knn_indices[:, 1:k + 1]  # 排除自身（假设自身距离最小） (M, k)

    # -------------------------- 3. 批量提取近邻并中心化 -------------------------
    # 提取所有近邻：(M, k, D)
    # 利用高级索引：knn_indices (M, k) -> 展平为 (M*k,)，索引X后重塑为 (M, k, D)
    neighbors = X[knn_indices.flatten()].reshape(M, k, D)

    # 中心化：每个邻域减去均值（批量操作）
    neighbors_centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # (M, k, D)

    # -------------------------- 4. 批量SVD分解（计算特征值） -------------------------
    # 对每个邻域的中心化矩阵进行SVD，取奇异值（按降序排列）
    # SVD返回：U (M, k, k), S (M, k), Vh (M, D, D)，此处仅需奇异值S
    S = svdvals(neighbors_centered)  # (M, k)

    # 计算特征值：特征值 = (奇异值^2) / (k-1)（样本协方差矩阵的特征值）
    eigenvalues = (S ** 2) / (k - 1)  # (M, k)

    # -------------------------- 5. 肘部法则确定局部维度 -------------------------
    def elbow_method(ev: torch.Tensor) -> int:
        """对单个样本的特征值应用肘部法则，返回局部维度"""
        ev = ev[ev > 1e-12]  # 过滤噪声（小特征值）
        if len(ev) <= 1:
            return len(ev)
            # 计算相邻特征值比值（后/前），找最大比值对应的"肘部"
        ratios = ev[1:] / ev[:-1]
        elbow_idx = torch.argmax(ratios).item()  # 最大比值位置
        return elbow_idx + 1  # 局部维度 = 肘部位置 + 1

    # 对所有M个样本应用肘部法则，收集局部维度
    local_dims = []
    for i in range(M):
        ev = eigenvalues[i]  # 当前样本的特征值 (k,)
        local_dim = elbow_method(ev)
        local_dims.append(local_dim)

        # -------------------------- 6. 中位数聚合局部维度 -------------------------
    return int(torch.median(torch.tensor(local_dims, device=device)).item())


def kde_manifold_mcmc_sampler(self,
    X: torch.Tensor,
    bw: float = 0.1,
    num_samples: int = 100,
    k_neighbors: int = 20,
    epsilon: float = 1e-3,
    sigma: float = 0.05,
    burn_in: int = 50,
    verbose: bool = False
) -> torch.Tensor:
    """
    基于KDE的低维流形梯度MCMC采样
    Args:
        X: 输入样本集，形状 (N, D)，N为样本数，D为原始维度
        bw: KDE核带宽（高斯核）
        manifold_dim: 低维流形维度（d < D）
        num_samples: 目标采样数量
        k_neighbors: 局部PCA的近邻数（用于切空间估计）
        epsilon: 黎曼梯度步长
        sigma: 切空间内噪声标准差
        burn_in: MCMC预热迭代次数（不计入最终采样结果）
        verbose: 是否打印采样进度
    Returns:
        采样结果，形状 (num_samples, D)
    """
    device = X.device
    N, D = X.shape
    samples = []

    def kde_log_pdf_grad(self, x: torch.Tensor) -> torch.Tensor:
        """计算KDE对数概率梯度，x形状 (D,)"""
        # 距离向量 (N, D)
        diff = x - X  # (N, D)
        dist_sq = torch.sum(diff ** 2, dim=1)  # (N,)
        # 高斯核梯度权重 (N,)
        kernel_weight = torch.exp(-dist_sq / (2 * bw ** 2)) / (bw ** 2)
        # 梯度 = 平均贡献 (D,)
        grad = torch.mean(kernel_weight.unsqueeze(1) * diff, dim=0) / (bw ** D * (2 * torch.pi) ** (D/2))
        # 对数梯度 = 梯度 / PDF（需重新计算PDF）
        pdf = torch.exp(self.mvn.log_prob(x))
        return grad / (pdf + 1e-12)  # 避免除零

    # -------------------------- 2. 流形切空间估计 -------------------------
    def estimate_tangent_space(x: torch.Tensor) -> torch.Tensor:
        """通过局部PCA估计x处的切空间正交基，返回 (D, d)"""
        # 找x的k近邻

        dist = torch.sum((x - X) ** 2, dim=-1)  # (N,)
        _, idx = torch.topk(dist, k=k_neighbors, largest=False)  # 最近邻索引
        neighbors = X[idx]  # (k_neighbors, D)
        # 局部PCA
        centered = neighbors - neighbors.mean(dim=0)  # 中心化
        cov = centered.T @ centered / k_neighbors  # 协方差矩阵 (D, D)
        eigvals, eigvecs = torch.linalg.eigh(cov)  # 特征值升序，特征向量按列排列
        # 取最大d个特征向量作为切空间基（注意eigh返回的特征向量需逆序）
        tangent_basis = eigvecs[:, -manifold_dim:]  # (D, d)
        return tangent_basis

    # -------------------------- 3. MCMC采样迭代 -------------------------
    # 初始化：从高概率区域选择初始点（KDE PDF最大的样本）
    manifold_dim = estimate_manifold_dimension_pytorch(X)
    with torch.no_grad():
        all_log_pdf = self.mvn.log_prob(X)  # 计算所有样本的KDE概率
        init_idx = torch.argmax(all_log_pdf)  # 最高概率样本索引
        current_x = X[init_idx].clone()  # 初始点

    # MCMC迭代
    for iter in range(burn_in + num_samples):
        # 估计当前点的切空间
        tangent_basis = estimate_tangent_space(current_x)  # (D, d)

        # 计算KDE对数概率梯度（欧氏梯度）
        grad_euclidean = kde_log_pdf_grad(self, current_x)  # (D,)

        # 黎曼梯度：欧氏梯度投影到切空间
        grad_riemannian = tangent_basis @ (tangent_basis.T @ grad_euclidean)  # (D,)

        # 提议分布：沿黎曼梯度方向 + 切空间噪声
        noise = torch.normal(0, sigma, size=(manifold_dim,), device=device)  # (d,)
        delta = epsilon * grad_riemannian + tangent_basis @ noise  # (D,)
        candidate_x = current_x + delta

        # Metropolis-Hastings接受准则
        log_p_current = self.mvn.log_prob(current_x)
        log_p_candidate = self.mvn.log_prob(candidate_x)
        accept_prob = torch.min(torch.tensor(1.0), torch.exp(log_p_candidate - log_p_current))

        if torch.rand(1, device=accept_prob.device) < accept_prob:
            current_x = candidate_x  # 接受候选点

        # 收集非预热样本
        if iter >= burn_in:
            samples.append(current_x.clone())
            if verbose and (len(samples) % 10 == 0):
                print(f"Sampled {len(samples)}/{num_samples} points")

    return torch.stack(samples), manifold_dim  # (num_samples, D)

from torch.linalg import svd

def precompute_anchors(anchors: torch.Tensor, kde_bw: float = 0.1, k_neighbors: int = 20, manifold_dim: int = 2):
    """
    预计算锚点的KDE-PDF值和局部切空间基（离线执行）
    Args:
        anchors: 锚点集，形状 (X, D)
        kde_bw: KDE带宽
        k_neighbors: 锚点局部近邻数（用于切空间估计）
        manifold_dim: 流形维度
    Returns:
        anchor_pdf: 锚点的PDF值，形状 (X,)
        anchor_tangent_basis: 锚点的切空间基，形状 (X, D, manifold_dim)
    """
    X, D = anchors.shape
    device = anchors.device

    # -------------------------- 预计算锚点KDE-PDF值 -------------------------
    # 批量计算锚点间距离矩阵 (X, X)
    dist_matrix_anchors = torch.cdist(anchors, anchors, p=2)
    # KDE核函数 (X, X)
    kernel_anchors = torch.exp(-dist_matrix_anchors ** 2 / (2 * kde_bw ** 2))
    # 锚点PDF值 (X,)
    anchor_pdf = torch.mean(kernel_anchors, dim=1)
    # anchor_pdf = anchor_pdf_unnormalized / torch.sum(anchor_pdf_unnormalized)
    # -------------------------- 预计算锚点切空间基 -------------------------
    anchor_tangent_basis = torch.zeros(X, D, manifold_dim, device=device)
    # 计算每个锚点的局部近邻
    _, knn_indices = torch.topk(dist_matrix_anchors, k=k_neighbors + 1, largest=False)  # (X, k+1)
    knn_indices = knn_indices[:, 1:k_neighbors + 1]  # 排除自身 (X, k_neighbors)

    # 批量提取近邻并中心化
    neighbors = anchors[knn_indices.flatten()].reshape(X, k_neighbors, D)  # (X, k_neighbors, D)
    neighbors_centered = neighbors - neighbors.mean(dim=1, keepdim=True)  # (X, k_neighbors, D)

    # 批量SVD分解求切空间基
    for i in range(X):
        _, _, V = svd(neighbors_centered[i])  # V: (D, D)，右奇异向量按列排列
        anchor_tangent_basis[i] = V[:, :manifold_dim]  # 取前manifold_dim个基向量

    return anchor_pdf, anchor_tangent_basis

class ManifoldClassifier:
    def __init__(self, class_manifolds: list, kde_bw: float = 0.1, alpha: float = 1.0):
        """
        基于流形表示的多类分类器
        Args:
            class_manifolds: 3个类别的流形参数列表，每个元素为precompute_class_manifold的输出
            kde_bw: KDE带宽（需与预计算时一致）
            alpha: 密度项在评分中的权重（平衡流形距离和密度）
        """
        self.class_manifolds = class_manifolds  # 3个类别的流形参数
        self.kde_bw = kde_bw
        self.alpha = alpha
        self.C = len(class_manifolds)  # 类别数=3
        self.D = class_manifolds[0]["anchors"].shape[1]  # 数据维度
        self.manifold_dim = class_manifolds[0]["anchor_tangent_basis"].shape[2]  # 流形维度

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        N = X.shape[0]
        D = X.shape[1]
        scores = []
        import numpy
        for c in range(self.C):
            anchors = self.class_manifolds[c]["anchors"]  # (X_c, D)，当前类别的锚点集
            anchor_tangent_basis = self.class_manifolds[c]["anchor_tangent_basis"]  # (X_c, D, manifold_dim)

            # -------------------------- 1. 计算样本到类别流形的距离 -------------------------
            dist_matrix = torch.cdist(X, anchors, p=2)  # (N, X_c)，样本与所有锚点的距离矩阵
            min_dist, nearest_idx = torch.min(dist_matrix, dim=1)  # (N,), (N,)，最近锚点距离和索引
            nearest_tangent = anchor_tangent_basis[nearest_idx]  # (N, D, manifold_dim)，最近锚点的切空间基
            X_centered = X - anchors[nearest_idx]  # (N, D)，样本相对于最近锚点的偏移

            # 批量矩阵乘法计算切空间投影（修复前序矩阵乘法错误）
            tangent_T = nearest_tangent.transpose(1, 2)  # (N, manifold_dim, D)
            X_centered_3d = X_centered.unsqueeze(2)  # (N, D, 1)
            tangent_T_x = torch.bmm(tangent_T, X_centered_3d)  # (N, manifold_dim, 1)
            proj_3d = torch.bmm(nearest_tangent, tangent_T_x)  # (N, D, 1)
            proj = proj_3d.squeeze(2)  # (N, D)，切空间投影结果
            manifold_dist = torch.sum((X_centered - proj) ** 2, dim=1)  # (N,)，流形距离

            # -------------------------- 2. 补充：计算样本在当前类别内的密度（pdf） -------------------------
            # KDE核函数：样本与所有锚点的核值 (N, X_c)
            max_dist = 3 * self.kde_bw  # 距离超过3倍带宽时，核函数≈0（经验值）
            kernel = torch.exp(-(dist_matrix ** 2) / (2 * self.kde_bw ** 2))
            kernel = torch.where(dist_matrix > max_dist, torch.zeros_like(kernel), kernel)  # 截断远距样本

            # 3. 移除分母，计算相对密度（单样本单值）
            pdf = torch.mean(kernel, dim=1)  # (N,)，每个样本对应一个密度值
            # Assuming kernel_anchors are log probabilities or similarities
            # log_kernel = torch.log(kernel)
            # log_pdf = torch.logsumexp(log_kernel, dim=1) - torch.log(
            #     torch.tensor(kernel.shape[1])) - 0.5 * D * torch.log(torch.tensor(2 * numpy.pi * self.kde_bw ** 2))
            #
            # # Convert to regular probabilities if needed
            # pdf = torch.exp(log_pdf - torch.max(log_pdf))
            # 4. 检查并替换可能的NaN/Inf（极端情况处理）
            pdf = torch.nan_to_num(pdf, nan=0.0, posinf=1e6, neginf=0.0)  # 用0替换NaN，1e6替换Inf
            # -------------------------- 3. 类别评分计算 -------------------------
            mu_m = torch.mean(manifold_dist)  # manifold_dist的均值
            std_m = torch.std(manifold_dist)  # manifold_dist的标准差
            mu_p = torch.mean(pdf)  # pdf的均值
            std_p = torch.std(pdf)  # pdf的标准差

            # -------------------------- 2. 分类分数计算（推理时） -------------------------
            # 标准化：(x - mu) / std，确保均值0、方差1
            manifold_dist_norm = (manifold_dist - mu_m) / (std_m + 1e-8)
            pdf_norm = (pdf - mu_p) / (std_p + 1e-8)

            # 此时两项尺度均为~1，alpha可设为1（或根据需求微调）
            class_score = -2*manifold_dist_norm + pdf_norm  # alpha=1即可平衡
            # class_score = -manifold_dist + self.alpha * pdf  # 综合流形距离和密度的评分
            scores.append(class_score)

        scores = torch.stack(scores, dim=1)  # (N, 3)
        proba = torch.softmax(scores, dim=1)  # (N, 3)
        return proba

    def predict(self, X: torch.Tensor):
        """预测类别标签（取概率最高的类别）"""
        proba = self.predict_proba(X)
        return torch.argmax(proba, dim=1), proba  # (N,)，类别索引0/1/2


class ImprovedManifoldClassifier:
    def __init__(self, class_manifolds: list, kde_bw: float = 0.1, alpha: float = 1.0,
                 beta: float = 0.5, use_attention: bool = True):
        self.class_manifolds = class_manifolds
        self.kde_bw = kde_bw
        self.alpha = alpha  # Density weight
        self.beta = beta  # Multiple anchors weight
        self.use_attention = use_attention
        self.C = len(class_manifolds)
        self.D = class_manifolds[0]["anchors"].shape[1]
        self.manifold_dim = class_manifolds[0]["anchor_tangent_basis"].shape[2]
        # Store device information
        self.device = class_manifolds[0]["anchors"].device
        # Precompute class statistics for better normalization
        self._precompute_class_stats()

    def _precompute_class_stats(self):
        """Precompute class-wise statistics with proper device handling"""
        self.class_stats = []
        for c in range(self.C):
            anchors = self.class_manifolds[c]["anchors"]
            anchor_pdf = self.class_manifolds[c]["anchor_pdf"]

            # Compute intra-class distances
            intra_dists = torch.pdist(anchors, p=2)

            # FIX: Ensure quantile tensor is on the same device
            quantiles = torch.tensor([0.1, 0.9], device=self.device)
            density_range = torch.quantile(anchor_pdf, quantiles)

            stats = {
                'intra_mean': torch.mean(intra_dists),
                'intra_std': torch.std(intra_dists),
                'anchor_scale': torch.norm(anchors, dim=1).mean(),
                'density_range': density_range
            }
            self.class_stats.append(stats)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        N = X.shape[0]
        scores = []

        for c in range(self.C):
            anchors = self.class_manifolds[c]["anchors"]
            anchor_tangent_basis = self.class_manifolds[c]["anchor_tangent_basis"]
            anchor_pdf = self.class_manifolds[c]["anchor_pdf"]

            dist_matrix = torch.cdist(X, anchors, p=2)

            if self.use_attention:
                attention_weights = self._compute_anchor_attention(dist_matrix, anchor_pdf)
                class_score = self._multi_anchor_scoring(X, anchors, anchor_tangent_basis,
                                                         anchor_pdf, dist_matrix, attention_weights,
                                                         class_idx=c)  # ADDED: pass class index
            else:
                class_score = self._single_anchor_scoring(X, anchors, anchor_tangent_basis,
                                                          anchor_pdf, dist_matrix, class_idx=c)  # Also fix this method

            scores.append(class_score)

        scores = torch.stack(scores, dim=1)
        proba = torch.softmax(scores, dim=1)
        return proba

    def _compute_anchor_attention(self, dist_matrix: torch.Tensor, anchor_pdf: torch.Tensor) -> torch.Tensor:
        """Compute attention weights over multiple anchors"""
        # Temperature-scaled attention: closer and higher density anchors get more weight
        temperature = 0.1 * self.kde_bw
        attention_scores = -dist_matrix / temperature + torch.log(anchor_pdf + 1e-8).unsqueeze(0)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return attention_weights

    def _multi_anchor_scoring(self, X: torch.Tensor, anchors: torch.Tensor,
                              tangent_bases: torch.Tensor, anchor_pdf: torch.Tensor,
                              dist_matrix: torch.Tensor, attention_weights: torch.Tensor,
                              class_idx: int) -> torch.Tensor:  # ADDED: class_idx parameter
        """Compute class score using multiple anchors with attention"""
        N, X_c = dist_matrix.shape
        manifold_dists = torch.zeros(N, device=X.device)
        density_scores = torch.zeros(N, device=X.device)

        # Compute weighted manifold distance and density over multiple anchors
        for i in range(X_c):
            anchor = anchors[i].unsqueeze(0)  # (1, D)
            tangent_basis = tangent_bases[i].unsqueeze(0)  # (1, D, manifold_dim)

            # Manifold distance for this anchor
            X_centered = X - anchor  # (N, D)
            tangent_T = tangent_basis.transpose(1, 2)  # (1, manifold_dim, D)
            X_centered_3d = X_centered.unsqueeze(2)  # (N, D, 1)
            tangent_T_x = torch.bmm(tangent_T.expand(N, -1, -1), X_centered_3d)  # (N, manifold_dim, 1)
            proj_3d = torch.bmm(tangent_basis.expand(N, -1, -1), tangent_T_x)  # (N, D, 1)
            proj = proj_3d.squeeze(2)  # (N, D)
            anchor_manifold_dist = torch.sum((X_centered - proj) ** 2, dim=1)  # (N,)

            # Density for this anchor (using precomputed anchor PDF)
            anchor_density = anchor_pdf[i]

            # Weight by attention
            anchor_weight = attention_weights[:, i]
            manifold_dists += anchor_weight * anchor_manifold_dist
            density_scores += anchor_weight * anchor_density

        # Use class-specific statistics
        stats = self.class_stats[class_idx]  # FIXED: Use class_idx instead of undefined c
        manifold_norm = manifold_dists / (stats['intra_mean'] + 1e-8)
        density_norm = density_scores / (stats['density_range'][1] + 1e-8)

        # Balanced scoring with class-specific adaptation
        class_score = -self.alpha * manifold_norm + self.beta * density_norm

        return class_score

    def _single_anchor_scoring(self, X: torch.Tensor, anchors: torch.Tensor,
                               tangent_bases: torch.Tensor, anchor_pdf: torch.Tensor,
                               dist_matrix: torch.Tensor, class_idx: int) -> torch.Tensor:  # ADDED: class_idx
        """Original single anchor scoring for comparison"""
        min_dist, nearest_idx = torch.min(dist_matrix, dim=1)
        nearest_tangent = tangent_bases[nearest_idx]
        X_centered = X - anchors[nearest_idx]

        # Projection
        tangent_T = nearest_tangent.transpose(1, 2)
        X_centered_3d = X_centered.unsqueeze(2)
        tangent_T_x = torch.bmm(tangent_T, X_centered_3d)
        proj_3d = torch.bmm(nearest_tangent, tangent_T_x)
        proj = proj_3d.squeeze(2)
        manifold_dist = torch.sum((X_centered - proj) ** 2, dim=1)

        # Density using KDE
        kernel = torch.exp(-(dist_matrix ** 2) / (2 * self.kde_bw ** 2))
        pdf = torch.mean(kernel, dim=1)
        pdf = torch.nan_to_num(pdf, nan=0.0, posinf=1e6, neginf=0.0)

        # Normalization using class-specific stats
        stats = self.class_stats[class_idx]  # FIXED: Use class_idx
        manifold_norm = manifold_dist / (stats['intra_mean'] + 1e-8)
        density_norm = pdf / (stats['density_range'][1] + 1e-8)

        return -self.alpha * manifold_norm + self.beta * density_norm

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        proba = self.predict_proba(X)
        return torch.argmax(proba, dim=1)

# Optimal parameters for different scenarios
MANIFOLD_CLASSIFIER_PARAMS = {
    # For well-separated manifolds
    'separated': {
        'kde_bw': 0.05,
        'alpha': 2.0,      # Strong manifold distance weighting
        'beta': 1.0,       # Moderate density weighting
        'use_attention': False,  # Single anchor sufficient
        'temperature': 0.8
    },
    # For overlapping manifolds
    'overlapping': {
        'kde_bw': 0.15,
        'alpha': 1.0,      # Balanced weighting
        'beta': 1.5,       # Higher density weighting
        'use_attention': True,   # Multiple anchors help
        'temperature': 1.2
    },
    # For high-dimensional data
    'high_dim': {
        'kde_bw': 0.1,
        'alpha': 0.8,      # Lower manifold weight (curse of dimensionality)
        'beta': 2.0,       # Higher density weight
        'use_attention': True,
        'temperature': 1.0
    }
}


class EnsembleManifoldClassifier:
    def __init__(self, class_manifolds: list, n_estimators: int = 5):
        self.classifiers = []
        for i in range(n_estimators):
            # Use different bandwidths and parameters for diversity
            bw = 1 + 0.1 * (i / (n_estimators - 1)) if n_estimators > 1 else 0.1
            alpha = 0.1 + 1.5 * (i / (n_estimators - 1)) if n_estimators > 1 else 1.0

            classifier = ImprovedManifoldClassifier(
                class_manifolds,
                kde_bw=bw,
                alpha=alpha,
                beta=10.0,
                use_attention=(i % 2 == 0),  # Mix of attention strategies
            )
            # classifier = ImprovedManifoldClassifier(
            #     class_manifolds,
            #     **MANIFOLD_CLASSIFIER_PARAMS['high_dim']
            # )
            self.classifiers.append(classifier)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        all_proba = []
        for classifier in self.classifiers:
            proba = classifier.predict_proba(X)
            all_proba.append(proba)

        # Average probabilities
        avg_proba = torch.stack(all_proba).mean(dim=0)
        return avg_proba

    def predict(self, X: torch.Tensor):
        proba = self.predict_proba(X)
        return torch.argmax(proba, dim=1), proba


def batch_pdf_manifold_loss(
        samples: torch.Tensor,  # 待优化样本集，形状 (n, D)
        anchors: torch.Tensor,  # 锚点集，形状 (X, D)
        anchor_pdf: torch.Tensor,  # 预计算的锚点PDF，形状 (X,)
        anchor_tangent_basis: torch.Tensor,  # 预计算的锚点切空间基，形状 (X, D, manifold_dim)
        kde_bw: float = 0.1,
        manifold_dim: int = 2,
        alpha: float = 20.0,
        beta: float = 10.0,
        epsilon: float = 1e-8
):
    """
    多样本多锚点场景下的批量PDF流形损失函数
    Args:
        samples: 输入样本集 (n, D)
        anchors: 锚点集 (X, D)
        anchor_pdf: 预计算的锚点PDF (X,)
        anchor_tangent_basis: 预计算的锚点切空间基 (X, D, manifold_dim)
    Returns:
        total_loss: 所有样本的平均损失
    """
    n, D = samples.shape
    X = anchors.shape[0]
    device = samples.device
    import numpy
    # -------------------------- 1. 批量计算样本KDE-PDF值 -------------------------
    # 样本-锚点距离矩阵 (n, X)
    dist_samples_anchors = torch.cdist(samples, anchors, p=2)
    # KDE核函数 (n, X)
    kernel_samples = torch.exp(-dist_samples_anchors ** 2 / (2 * kde_bw ** 2))
    # 样本PDF值 (n,)
    sample_pdf = torch.mean(kernel_samples, dim=1) #/ (kde_bw ** D * (2 * torch.pi) ** (D / 2))
    # log_kernel = torch.log(kernel_samples)
    # log_pdf = torch.logsumexp(log_kernel, dim=1) - torch.log(
    #     torch.tensor(kernel_samples.shape[1])) - 0.5 * D * torch.log(torch.tensor(2 * numpy.pi * kde_bw ** 2))
    # # Convert to regular probabilities if needed
    # sample_pdf = torch.exp(log_pdf - torch.max(log_pdf))
    # sample_pdf = sample_pdf / torch.sum(sample_pdf)
    # -------------------------- 2. 批量寻找最近锚点 -------------------------
    # 每个样本的最近锚点索引 (n,)
    _, nearest_anchor_idx = torch.min(dist_samples_anchors, dim=1)
    # 最近锚点的PDF值 (n,)
    nearest_anchor_pdf = anchor_pdf[nearest_anchor_idx]
    # 最近锚点的切空间基 (n, D, manifold_dim)
    nearest_tangent_basis = anchor_tangent_basis[nearest_anchor_idx]

    # -------------------------- 3. 批量计算样本权重 -------------------------
    sample_weights = 1.0 / (sample_pdf + epsilon)  # (n,)，稀疏样本权重更高

    # -------------------------- 4. 批量计算锚点距离项 -------------------------
    # 样本到最近锚点的距离平方 (n,)
    nearest_dist_sq = torch.gather(dist_samples_anchors ** 2, dim=1,
                                   index=nearest_anchor_idx.unsqueeze(1)).squeeze()
    distance_loss = torch.mean(sample_weights * nearest_anchor_pdf * nearest_dist_sq)

    # -------------------------- 5. 批量计算PDF对齐项 -------------------------
    pdf_alignment_loss = alpha * torch.mean(sample_weights * (nearest_anchor_pdf - sample_pdf) ** 2)

    # -------------------------- 6. 批量计算流形正则项 -------------------------
    # 样本中心化（相对于最近锚点的近邻）
    samples_centered = samples - anchors[nearest_anchor_idx]  # (n, D)
    # 样本投影到最近锚点的切空间 (n, D)
    # samples_proj = nearest_tangent_basis @ (
    #             nearest_tangent_basis.transpose(1, 2) @ samples_centered.unsqueeze(2)).squeeze(2)
    # 流形距离平方 (n,)
    # 步骤1：计算括号内部分：(N, K, D) @ (N, D, 1) → (N, K, 1)
    tangent_T = nearest_tangent_basis.transpose(1, 2)  # 转置：(N, K, D)
    samples_centered_3d = samples_centered.unsqueeze(2)  # 升维：(N, D, 1)
    intermediate = torch.bmm(tangent_T, samples_centered_3d)  # 批量乘法：(N, K, 1)

    # 步骤2：计算投影：(N, D, K) @ (N, K, 1) → (N, D, 1) → 降维为(N, D)
    samples_proj = torch.bmm(nearest_tangent_basis, intermediate).squeeze(2)  # 最终投影


    manifold_dist_sq = torch.sum((samples_centered - samples_proj) ** 2, dim=1)
    manifold_reg_loss = beta * torch.mean(manifold_dist_sq)

    # 总损失（平均损失）
    total_loss = distance_loss + pdf_alignment_loss + manifold_reg_loss
    return total_loss
def adaptive_weighting(sample_pdf, nearest_anchor_pdf, strategy="inverse_pdf"):
    if strategy == "inverse_pdf":
        return 1.0 / (sample_pdf + 1e-8)
    elif strategy == "exponential":
        # Less aggressive than inverse
        return torch.exp(-sample_pdf * 10.0)
    elif strategy == "soft_inverse":
        return 1.0 / (sample_pdf + 0.1)  # Less extreme
    elif strategy == "contrastive":
        # Focus on moderate density regions
        return torch.abs(sample_pdf - 0.5) + 0.5
    else:
        return torch.ones_like(sample_pdf)


def multi_scale_kde(samples, anchors, bandwidths=[0.05, 0.1, 0.2]):
    multi_sample_pdf = []
    for bw in bandwidths:
        dist_matrix = torch.cdist(samples, anchors, p=2)
        kernel = torch.exp(-dist_matrix ** 2 / (2 * bw ** 2))
        pdf = torch.mean(kernel, dim=1)
        multi_sample_pdf.append(pdf)

    # Combine multi-scale PDFs
    sample_pdf = torch.stack(multi_sample_pdf).mean(dim=0)
    return sample_pdf


def robust_distances(samples, anchors):
    # L2 distance with clipping to avoid outliers
    raw_dist = torch.cdist(samples, anchors, p=2)

    # Robust distance 1: Clipped distance
    dist_clipped = torch.clamp(raw_dist, max=3.0)  # Clip extreme distances

    # Robust distance 2: Huber-like distance
    delta = 1.0
    huber_dist = torch.where(
        raw_dist < delta,
        0.5 * raw_dist ** 2,
        delta * (raw_dist - 0.5 * delta)
    )

    return huber_dist  # More robust to outliers


def enhanced_manifold_regularization(samples, anchors, tangent_basis, beta=10.0, gamma=1.0):
    # Original manifold regularization
    samples_centered = samples - anchors
    tangent_T = tangent_basis.transpose(1, 2)
    intermediate = torch.bmm(tangent_T, samples_centered.unsqueeze(2))
    samples_proj = torch.bmm(tangent_basis, intermediate).squeeze(2)
    manifold_dist_sq = torch.sum((samples_centered - samples_proj) ** 2, dim=1)

    # Additional smoothness regularization
    # Encourage smooth transitions between tangent spaces
    if tangent_basis.shape[0] > 1:  # Multiple anchors
        # Compute consistency between neighboring tangent spaces
        tangent_similarity = torch.bmm(
            tangent_basis[:-1].transpose(1, 2),
            tangent_basis[1:]
        )
        tangent_consistency = torch.mean(1.0 - torch.abs(tangent_similarity))
    else:
        tangent_consistency = torch.tensor(0.0)

    return beta * torch.mean(manifold_dist_sq) + gamma * tangent_consistency


def improved_batch_pdf_manifold_loss(
        samples: torch.Tensor,
        anchors: torch.Tensor,
        anchor_pdf: torch.Tensor,
        anchor_tangent_basis: torch.Tensor,
        kde_bw: float = 1,
        manifold_dim: int = 2,
        alpha: float = 20.0,
        beta: float = 15.0,
        gamma: float = 1.0,  # New: consistency regularization
        delta: float = 0.1,  # New: diversity regularization
        epsilon: float = 1e-8,
        compactness_lambda: float = 30.0,    # Controls intra-cluster compactness
        separation_mu: float = 1.0,         # Controls inter-cluster separation
        smoothness_nu: float = 1,         # Controls manifold smoothness
):
    n, D = samples.shape
    device = samples.device



    # 1. Multi-scale KDE for better density estimation
    sample_pdf = multi_scale_kde(samples, anchors, bandwidths=[kde_bw * 0.5, kde_bw, kde_bw * 2.0])

    # 2. Robust distance computation
    dist_samples_anchors = robust_distances(samples, anchors)

    # 3. Adaptive weighting
    sample_weights = adaptive_weighting(sample_pdf, anchor_pdf, strategy="soft_inverse")

    # 4. Find nearest anchors
    _, nearest_anchor_idx = torch.min(dist_samples_anchors, dim=1)
    nearest_anchor_pdf = anchor_pdf[nearest_anchor_idx]
    nearest_tangent_basis = anchor_tangent_basis[nearest_anchor_idx]

    # NEW: Intra-cluster compactness loss
    anchor_centers = anchors[nearest_anchor_idx]
    compactness_loss = compactness_lambda * torch.mean(
        sample_weights * torch.norm(samples - anchor_centers, dim=1) ** 2
    )
    # NEW: Manifold smoothness regularization
    tangent_consistency = smoothness_nu * _compute_tangent_consistency(
        nearest_tangent_basis, nearest_anchor_idx
    )

    # NEW: Inter-cluster separation loss
    # 10. NEW: Simple inter-cluster separation loss
    if n > 1:
        separation_loss = simple_separation_loss(samples, nearest_anchor_idx, separation_mu)
    else:
        separation_loss = torch.tensor(0.0, device=device)

    # 5. Distance loss with temperature scaling
    nearest_dist_sq = torch.gather(dist_samples_anchors, dim=1,
                                   index=nearest_anchor_idx.unsqueeze(1)).squeeze()

    # Temperature-scaled distance loss
    temperature = 1.0 / (nearest_anchor_pdf + epsilon)
    distance_loss = torch.mean(sample_weights * temperature * nearest_dist_sq)

    # 6. PDF alignment with margin
    pdf_diff = nearest_anchor_pdf - sample_pdf
    # Add margin to prevent overfitting to exact PDF matching
    pdf_alignment_loss = alpha * torch.mean(sample_weights * torch.clamp(pdf_diff ** 2 - 0.01, min=0))

    # 7. Enhanced manifold regularization
    manifold_reg_loss = enhanced_manifold_regularization(
        samples, anchors[nearest_anchor_idx], nearest_tangent_basis, beta, gamma
    )
    # 8. Additional: Diversity regularization (prevents mode collapse)
    if n > 1:
        sample_diversity = -torch.pdist(samples, p=2).mean()  # Encourage spread
    else:
        sample_diversity = torch.tensor(0.0)

    # total_loss = (distance_loss +
    #               pdf_alignment_loss +
    #               manifold_reg_loss +
    #               delta * sample_diversity)

    total_loss = (distance_loss + pdf_alignment_loss + 0.1*manifold_reg_loss +
                  0.1 * sample_diversity + 10*compactness_loss +
                  separation_loss + 0.1*tangent_consistency)

    return total_loss


def simple_separation_loss(samples: torch.Tensor, nearest_anchor_idx: torch.Tensor,
                           separation_mu: float, min_separation: float = 0.5):
    """Simplified separation loss that's more numerically stable"""
    n = samples.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=samples.device)

    # Compute pairwise distances between all samples
    pairwise_dists = torch.pdist(samples, p=2)  # Shape: (n*(n-1)/2,)

    # Create mask for samples from different anchors
    anchor_mask = []
    for i in range(n):
        for j in range(i + 1, n):
            if nearest_anchor_idx[i] != nearest_anchor_idx[j]:
                anchor_mask.append(1.0)
            else:
                anchor_mask.append(0.0)

    anchor_mask = torch.tensor(anchor_mask, device=samples.device)

    if anchor_mask.sum() == 0:
        return torch.tensor(0.0, device=samples.device)

    # Penalize small distances between samples from different anchors
    separation_penalty = torch.relu(min_separation - pairwise_dists) * anchor_mask
    separation_loss = separation_mu * torch.sum(separation_penalty) / (anchor_mask.sum() + 1e-8)

    return separation_loss

def _compute_tangent_consistency(tangent_bases, anchor_indices):
    """Encourage smooth transitions between neighboring tangent spaces"""
    unique_anchors = torch.unique(anchor_indices)
    if len(unique_anchors) < 2:
        return torch.tensor(0.0)

    # Compute consistency between neighboring anchor tangent spaces
    consistency_loss = 0.0
    count = 0

    for i in range(len(unique_anchors) - 1):
        anchor1 = unique_anchors[i]
        anchor2 = unique_anchors[i + 1]

        basis1 = tangent_bases[anchor_indices == anchor1][0]
        basis2 = tangent_bases[anchor_indices == anchor2][0]

        # Compute subspace similarity
        similarity = torch.norm(basis1.T @ basis2, p='fro')
        consistency_loss += (1.0 - similarity) ** 2
        count += 1

    return consistency_loss / count if count > 0 else torch.tensor(0.0)


# Progressive difficulty scheduling
def get_dynamic_parameters(epoch, max_epochs):
    """Dynamically adjust parameters during training"""
    progress = epoch / max_epochs

    # Start with stronger manifold regularization, then focus on PDF alignment
    alpha = 50.0 * min(1.0, progress * 2)  # Ramp up PDF alignment
    beta = 20.0 * max(0.5, 1.0 - progress * 0.5)  # Reduce manifold emphasis

    # Start with larger bandwidth, then refine
    kde_bw = 0.2 * (1.0 - progress * 0.5) + 0.05

    return alpha, beta, kde_bw


# ### **关键模块说明**
# #### 1. **KDE概率密度估计**
# - **核心思想**：通过高斯核对每个样本的局部贡献加权求和，估计任意点的概率密度。
# - **数值稳定性**：使用`log-sum-exp`思想避免指数溢出，并添加极小值（`1e-12`）防止`log(0)`。
#
# #### 2. **流形切空间估计**
# - **局部PCA**：对当前采样点的`k_neighbors`个近邻点进行PCA，取最大`manifold_dim`个特征向量作为切空间基，捕捉流形局部线性结构。
# - **几何约束**：确保采样扰动仅沿切空间方向，避免垂直于流形的无效探索。
#
# #### 3. **黎曼梯度引导**
# - **梯度投影**：将KDE对数概率的欧氏梯度投影到切空间，得到沿流形表面的梯度方向，引导采样向高概率区域移动。
#
# #### 4. **MCMC平衡探索与利用**
# - **提议分布**：结合梯度方向（利用高概率区域）和随机噪声（探索流形结构），平衡“ exploitation”与“exploration”。
# - **预热迭代**：前`burn_in`次迭代用于让链收敛到目标分布，不计入最终采样结果。


### **使用示例**
# ```python
# 生成测试数据（2D流形嵌入到3D空间）

# N = 500
# t = torch.linspace(0, 4*torch.pi, N)
# X = torch.stack([
#     t * torch.cos(t),
#     t * torch.sin(t),
#     t / 5
# ], dim=1)  # 螺旋线流形 (N, 3)
#
# # 采样
# samples = kde_manifold_mcmc_sampler(
#     X=X,
#     bw=0.5,
#     num_samples=100,
#     k_neighbors=20,
#     verbose=True
# )
# print("采样结果形状:", samples.shape)  # (100, 3)

def occasional_mode_jump(current_x: torch.Tensor, X: torch.Tensor,
                        jump_prob: float = 0.01):
    """Occasionally jump to new high-density regions"""
    if torch.rand(1) < jump_prob:
        with torch.no_grad():
            # Find under-explored high-density regions
            current_samples = torch.stack(samples) if samples else current_x.unsqueeze(0)
            if len(current_samples) > 10:
                # Compute sample density coverage
                sample_density = torch.exp(kde_log_pdf(current_samples))
                # Find high-density points in X that are far from current samples
                dist_to_samples = torch.cdist(X, current_samples).min(dim=1)[0]
                combined_score = torch.exp(kde_log_pdf(X)) * dist_to_samples
                new_idx = torch.argmax(combined_score)
                return X[new_idx].clone()
    return current_x


def manifold_aware_proposal(kde, current_x: torch.Tensor, X: torch.Tensor,
                            tangent_basis: torch.Tensor, epsilon: float,
                            sigma: float, exploration_factor: float = 0.1):
    """
    Improved proposal that balances manifold following and exploration
    """
    device = current_x.device

    # Ensure proper shapes
    current_x = current_x.flatten()  # Ensure (D,)
    D = current_x.shape[0]

    # Get gradient and ensure proper shape
    grad_euclidean = kde.log_pdf_grad(current_x)
    grad_euclidean = grad_euclidean.flatten()  # Ensure (D,)

    manifold_dim = tangent_basis.shape[1]

    # Riemannian gradient projection
    grad_riemannian = tangent_basis @ (tangent_basis.T @ grad_euclidean)

    # Exploration term (orthogonal to manifold)
    if manifold_dim < D:
        # Method 1: Use SVD to find orthogonal complement (more reliable)
        try:
            # Perform SVD on tangent basis
            U, S, Vh = torch.linalg.svd(tangent_basis, full_matrices=True)
            # The last (D - manifold_dim) columns of U form the orthogonal complement
            orthogonal_basis = U[:, manifold_dim:]

            # Verify we got the right shape
            if orthogonal_basis.shape[1] != (D - manifold_dim):
                # If SVD didn't give full complement, use alternative method
                orthogonal_basis = _compute_orthogonal_complement_fallback(tangent_basis, D, manifold_dim, device)
        except:
            # Fallback if SVD fails
            orthogonal_basis = _compute_orthogonal_complement_fallback(tangent_basis, D, manifold_dim, device)

        # Double-check the shape
        if orthogonal_basis.shape[1] > 0:
            ortho_noise = torch.normal(0, exploration_factor * sigma,
                                       size=(orthogonal_basis.shape[1],), device=device)
            exploration_term = orthogonal_basis @ ortho_noise
        else:
            exploration_term = torch.zeros(D, device=device)
    else:
        exploration_term = torch.zeros(D, device=device)

    # Tangent space noise
    tangent_noise = torch.normal(0, sigma, size=(manifold_dim,), device=device)
    tangent_term = tangent_basis @ tangent_noise

    delta = epsilon * grad_riemannian + tangent_term + exploration_term
    return delta


def _compute_orthogonal_complement_fallback(tangent_basis: torch.Tensor, D: int,
                                            manifold_dim: int, device: torch.device) -> torch.Tensor:
    """
    Fallback method to compute orthogonal complement when SVD/QR fails
    """
    # Method 1: Use Gram-Schmidt to build orthogonal complement
    try:
        # Start with identity and remove tangent space components
        complement_basis = []

        # Generate candidate vectors
        for i in range(D):
            candidate = torch.zeros(D, device=device)
            candidate[i] = 1.0

            # Project out tangent space components
            for j in range(manifold_dim):
                candidate = candidate - tangent_basis[:, j] * (tangent_basis[:, j] @ candidate)

            # Normalize if not zero
            norm = torch.norm(candidate)
            if norm > 1e-8:
                complement_basis.append(candidate / norm)

            if len(complement_basis) >= (D - manifold_dim):
                break

        if complement_basis:
            orthogonal_basis = torch.stack(complement_basis, dim=1)
            return orthogonal_basis
        else:
            return torch.zeros(D, 0, device=device)

    except:
        # Final fallback: return empty (no exploration)
        return torch.zeros(D, 0, device=device)


def estimate_tangent_space_improved(x: torch.Tensor, X: torch.Tensor,
                                    k_neighbors: int, manifold_dim: int) -> torch.Tensor:
    """
    Improved tangent space estimation with better error handling
    """
    device = x.device

    # Ensure proper shape
    x = x.flatten()
    D = x.shape[0]

    try:
        # Find k nearest neighbors (more robust distance computation)
        distances = torch.cdist(x.unsqueeze(0), X, p=2).squeeze(0)

        # Handle case where k_neighbors might be too large
        actual_k = min(k_neighbors, X.shape[0])
        _, idx = torch.topk(distances, k=actual_k, largest=False)
        neighbors = X[idx]  # (actual_k, D)

        # Check if we have enough neighbors for stable PCA
        if actual_k < manifold_dim + 1:
            # Return random basis if not enough neighbors
            return torch.randn(D, manifold_dim, device=device)

        # Local PCA with regularization
        centered = neighbors - neighbors.mean(dim=0)

        # Regularize covariance matrix for stability
        cov = centered.T @ centered / actual_k
        cov_reg = cov + 1e-6 * torch.eye(D, device=device)  # Add small regularization

        # Eigen decomposition with fallback
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov_reg)
            # Take top manifold_dim eigenvectors
            if eigvecs.shape[1] >= manifold_dim:
                tangent_basis = eigvecs[:, -manifold_dim:]  # (D, manifold_dim)
            else:
                # Pad with random vectors if not enough eigenvectors
                tangent_basis = torch.randn(D, manifold_dim, device=device)
        except:
            # Fallback to random basis
            tangent_basis = torch.randn(D, manifold_dim, device=device)

        # Orthonormalize the basis
        try:
            tangent_basis, _ = torch.linalg.qr(tangent_basis)
        except:
            # Normalize columns if QR fails
            norms = torch.norm(tangent_basis, dim=0, keepdim=True)
            tangent_basis = tangent_basis / (norms + 1e-8)

        return tangent_basis

    except Exception as e:
        # Ultimate fallback
        print(f"Tangent space estimation failed: {e}")
        return torch.randn(D, manifold_dim, device=device)


def manifold_aware_proposal1(kde, current_x: torch.Tensor, X: torch.Tensor,
                            tangent_basis: torch.Tensor, epsilon: float,
                            sigma: float, exploration_factor: float = 0.1):
    """
    Improved proposal that balances manifold following and exploration
    """
    # Gradient term (manifold following)

    grad_euclidean = kde.log_pdf_grad(current_x)
    grad_riemannian = tangent_basis @ (tangent_basis.T @ grad_euclidean)


    # Exploration term (orthogonal to manifold)
    D = current_x.shape[0]
    manifold_dim = tangent_basis.shape[1]

    if manifold_dim < D:
        # Find orthogonal complement
        U, S, V = torch.svd(tangent_basis)
        orthogonal_basis = U[:, manifold_dim:]  # Orthogonal directions

        # Add exploration in orthogonal directions
        ortho_noise = torch.normal(0, exploration_factor * sigma,
                                   size=(D - manifold_dim,), device=device)
        exploration_term = orthogonal_basis @ ortho_noise
    else:
        exploration_term = torch.zeros_like(current_x)

    # Tangent space noise
    tangent_noise = torch.normal(0, sigma, size=(manifold_dim,), device=device)
    tangent_term = tangent_basis @ tangent_noise

    delta = epsilon * grad_riemannian + tangent_term + exploration_term
    return delta


def annealed_kde_log_pdf(x: torch.Tensor, X: torch.Tensor, bw: float,
                         temperature: float = 1.0) -> torch.Tensor:
    """Temperature-annealed KDE for better exploration"""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    dist_sq = torch.sum((x.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=-1)
    kernel = torch.exp(-dist_sq / (2 * bw ** 2))
    pdf = torch.mean(kernel, dim=1)

    # Apply temperature scaling
    log_pdf = torch.log(pdf + 1e-12)
    return log_pdf / temperature


def adaptive_sampling_parameters(
        X: torch.Tensor,
        current_x: torch.Tensor,
        iteration: int,
        max_iterations: int,
        base_bw: float,  # Add base bandwidth parameter
        base_epsilon: float,  # Add base step size parameter
        acceptance_rate: float = None  # Make acceptance rate explicit
):
    """Adapt parameters during sampling
    Args:
        X: Input samples for reference
        current_x: Current sample position
        iteration: Current iteration number
        max_iterations: Total iterations
        base_bw: Initial bandwidth
        base_epsilon: Initial step size
        acceptance_rate: Current acceptance rate (optional)
    """
    progress = iteration / max_iterations

    # Strategy 1: Anneal bandwidth for finer exploration
    # Start with larger BW for exploration, then refine
    adaptive_bw = base_bw * (1.0 + 0.8 * (1.0 - progress))

    # Strategy 2: Adaptive step size based on acceptance rate
    if acceptance_rate is not None:
        target_rate = 0.234  # Optimal for random walk Metropolis
        rate_diff = acceptance_rate - target_rate
        # Adjust step size: decrease if acceptance too high, increase if too low
        adaptive_epsilon = base_epsilon * torch.exp(torch.tensor(rate_diff))
    else:
        # Strategy 3: Progressive step size reduction
        adaptive_epsilon = base_epsilon * (1.0 - 0.5 * progress)

    # Strategy 4: Local density-aware adjustment
    with torch.no_grad():
        # Estimate local density around current point
        local_density = torch.exp(annealed_kde_log_pdf(current_x, X, base_bw))
        # In high density: smaller steps for precision
        # In low density: larger steps for exploration
        density_factor = 1.0 + 2.0 * (0.5 - local_density)  # [0.5, 1.5] range
        adaptive_epsilon = adaptive_epsilon * density_factor

    return adaptive_bw, adaptive_epsilon


# Track acceptance rate
acceptance_history = []


def select_diverse_samples(samples: torch.Tensor, k: int) -> torch.Tensor:
    """Select k diverse samples using farthest point sampling"""
    selected = [samples[0]]
    distances = torch.cdist(samples, samples[0:1]).squeeze()

    for _ in range(1, k):
        idx = torch.argmax(distances)
        selected.append(samples[idx])
        new_dists = torch.cdist(samples, samples[idx:idx + 1]).squeeze()
        distances = torch.minimum(distances, new_dists)

    return torch.stack(selected)


def manifold_aware_proposal_robust(kde, current_x: torch.Tensor, X: torch.Tensor,
                                   tangent_basis: torch.Tensor, epsilon: float,
                                   sigma: float, exploration_factor: float = 0.1):
    """
    More robust version with comprehensive dimension validation
    """
    device = current_x.device

    # Ensure proper shapes
    current_x = current_x.flatten()
    D = current_x.shape[0]

    grad_euclidean = kde.log_pdf_grad(current_x).flatten()
    manifold_dim = tangent_basis.shape[1]

    # Validate dimensions
    if tangent_basis.shape[0] != D:
        raise ValueError(f"Tangent basis dimension mismatch: {tangent_basis.shape[0]} != {D}")

    if grad_euclidean.shape[0] != D:
        raise ValueError(f"Gradient dimension mismatch: {grad_euclidean.shape[0]} != {D}")

    # Riemannian gradient
    grad_riemannian = tangent_basis @ (tangent_basis.T @ grad_euclidean)

    # Exploration term with safe orthogonal complement
    exploration_term = _safe_orthogonal_exploration(
        tangent_basis, D, manifold_dim, sigma, exploration_factor, device
    )

    # Tangent space noise
    tangent_noise = torch.normal(0, sigma, size=(manifold_dim,), device=device)
    tangent_term = tangent_basis @ tangent_noise

    delta = epsilon * grad_riemannian + tangent_term + exploration_term

    return delta


def _safe_orthogonal_exploration(tangent_basis: torch.Tensor, D: int, manifold_dim: int,
                                 sigma: float, exploration_factor: float, device: torch.device) -> torch.Tensor:
    """
    Safely compute exploration term in orthogonal complement
    """
    if manifold_dim >= D:
        return torch.zeros(D, device=device)

    try:
        # Method 1: SVD with full matrices (most reliable)
        U, S, Vh = torch.linalg.svd(tangent_basis, full_matrices=True)
        orthogonal_basis = U[:, manifold_dim:]

        # Verify we have the expected number of dimensions
        expected_ortho_dims = D - manifold_dim
        if orthogonal_basis.shape[1] != expected_ortho_dims:
            # If SVD returned wrong size, use projection method
            orthogonal_basis = _compute_orthogonal_by_projection(tangent_basis, D, manifold_dim, device)

        # Final safety check
        if orthogonal_basis.shape[1] == 0:
            return torch.zeros(D, device=device)

        # Generate noise and compute exploration
        ortho_noise = torch.normal(0, exploration_factor * sigma,
                                   size=(orthogonal_basis.shape[1],), device=device)
        return orthogonal_basis @ ortho_noise

    except Exception as e:
        print(f"Orthogonal exploration failed: {e}")
        return torch.zeros(D, device=device)


def _compute_orthogonal_by_projection(tangent_basis: torch.Tensor, D: int,
                                      manifold_dim: int, device: torch.device) -> torch.Tensor:
    """
    Compute orthogonal complement using projection matrix
    """
    try:
        # Projection matrix onto tangent space
        P_tangent = tangent_basis @ tangent_basis.T

        # Projection matrix onto orthogonal complement
        P_ortho = torch.eye(D, device=device) - P_tangent

        # Find basis for orthogonal complement by SVD of projection matrix
        U_ortho, S_ortho, Vh_ortho = torch.linalg.svd(P_ortho)

        # The number of non-zero singular values gives the dimension
        ortho_rank = torch.sum(S_ortho > 1e-6).item()
        orthogonal_basis = U_ortho[:, :ortho_rank]

        return orthogonal_basis

    except:
        return torch.zeros(D, 0, device=device)


def _mode_hop(kde, X: torch.Tensor, current_samples: list, current_x: torch.Tensor):
    """Jump to under-explored high-density region"""
    with torch.no_grad():
        if len(current_samples) > 10:
            # Find high-density regions far from current samples
            sample_tensor = torch.stack(current_samples[-10:])
            density_scores = torch.exp(kde.log_pdf(X))

            # Compute minimum distance to recent samples
            min_dists = torch.cdist(X, sample_tensor).min(dim=1)[0]

            # Combined score: high density + far from current samples
            combined_scores = density_scores * min_dists
            new_idx = torch.argmax(combined_scores)
            return X[new_idx].clone()

    return current_x  # Fallback to current position


def select_diverse_samples_improved(samples: torch.Tensor, k: int,
                                    density_weights: torch.Tensor = None):
    """Select samples considering both diversity and density"""
    if density_weights is None:
        density_weights = torch.ones(len(samples))

    selected = []
    remaining_indices = list(range(len(samples)))

    # Start with highest density sample
    start_idx = torch.argmax(density_weights)
    selected.append(samples[start_idx])
    remaining_indices.remove(start_idx.item())

    # Greedy selection: balance diversity and density
    while len(selected) < k and remaining_indices:
        max_score = -1
        best_idx = -1

        for idx in remaining_indices:
            candidate = samples[idx]
            # Minimum distance to selected samples
            min_dist = min([torch.norm(candidate - s) for s in selected])
            # Combined score: distance * density
            score = min_dist * density_weights[idx]

            if score > max_score:
                max_score = score
                best_idx = idx

        if best_idx >= 0:
            selected.append(samples[best_idx])
            remaining_indices.remove(best_idx)

    return torch.stack(selected)


def improved_adaptive_parameters(
        X: torch.Tensor, current_x: torch.Tensor, iteration: int,
        max_iterations: int, base_bw: float, base_epsilon: float,
        acceptance_rate: float = None, target_rate: float = 0.234
):
    progress = iteration / max_iterations

    # Adaptive bandwidth: start broad, then refine
    if progress < 0.3:  # Exploration phase
        adaptive_bw = base_bw * 1.5
    elif progress < 0.7:  # Transition phase
        adaptive_bw = base_bw
    else:  # Refinement phase
        adaptive_bw = base_bw * 0.7

    # Adaptive step size based on acceptance rate
    if acceptance_rate is not None:
        rate_ratio = acceptance_rate / target_rate
        # More aggressive adjustment
        adaptive_epsilon = base_epsilon * torch.exp(2.0 * (rate_ratio - 1.0))
        # Clamp to reasonable range
        adaptive_epsilon = torch.clamp(adaptive_epsilon, base_epsilon * 0.1, base_epsilon * 5.0)
    else:
        # Progressive refinement
        adaptive_epsilon = base_epsilon * (1.0 - 0.6 * progress)

    return adaptive_bw, adaptive_epsilon