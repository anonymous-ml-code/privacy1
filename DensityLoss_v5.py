import torch
from torch.distributions import MultivariateNormal, Normal, LowRankMultivariateNormal
from torch.distributions.distribution import Distribution
# from sklearn.datasets import make_blobs
# from torch.optim import LBFGS
import numpy as np
import math
from Manifold_Sampling1 import kde_manifold_mcmc_sampler, enhanced_manifold_mcmc_sampler
class GaussianKDE(Distribution):
    def __init__(self, X, bw=0.1, low=-25000, high=-1, base =1e100, temperature=1):
        """
        # -450000 is better for the training samples with width 0.9 and dimension 196

        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X.cuda()
        self.bw = bw
        self.low = low
        self.high = high
        self.low_test = 0
        self.low_train = 0
        self.high_train = 0
        self.high_test = 0
        self.temperature = temperature
        # self.bw = torch.nn.Parameter(torch.tensor(0.1)).cuda()

        # torch.tensor(bw, dtype=torch.double)
        # if X.device == 'cuda:0':
        #     self.bw = self.bw.cuda()
        # self.dims = 8
        self.dims = X.shape[-1]
        # self.base = torch.tensor(self.bw ** (-self.dims), dtype=torch.double)
        self.n = X.shape[0]
        # mean = torch.mean(X, dim=0)
        self.delta = 0.00001
        X = X.cuda()
        mean = X.mean(dim=0).cuda()
        # mean = torch.mean(X, dim=0)
        # self.m = mean
        fs_m = X.sub(mean.expand_as(X))
        cov_mat = fs_m.transpose(0, 1).mm(fs_m) / (self.n - 1)
        # #
        noise = self.delta * torch.eye(cov_mat.shape[0]).cuda().double()

        if X.device == 'cuda:0':
            noise = noise.cuda()
        # cov = torch.cov(X)
        cov = cov_mat.cuda() + noise
        # cov = torch.eye(X.shape[1])
        if X.shape[0] < 100:
            cov = torch.eye(X.shape[1]).cuda()
        # cov = torch.cov(fs_m)
        # self.c = cov
        # self.mean = mean
        # self.cov = cov

        # X_ = X - mean
        # cov = torch.matmul(X_.T, X_)/ (X_.shape[0]-1)
        # cov = torch.cov(X)
        # cov = cov + 1e-6 * cov.mean() * torch.rand_like(cov)
        # epsilon = 0.000001
        # Add small pertturbation.
        # cov = cov + epsilon * torch.eye(self.dims).double().cuda()
        # s = torch.diag(s)
        cov = torch.where(cov == -float('inf'), torch.tensor(float('-1e6')), cov)
        cov = torch.where(cov == float('inf'), torch.tensor(float('1e6')), cov)

        self.mvn = MultivariateNormal(loc=mean, covariance_matrix=cov)

        # self.mvn = MultivariateNormal(loc=mean, scale_tril=torch.tril(cov))

    def sample1(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        # s = torch.cov(self.X[idxs].transpose(0, 1)).diag().sqrt()
        # norm = Normal(loc=self.X[idxs], scale=self.bw)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    # @staticmethod
    def sample_mcmc(self, num_samples, X, bw):
        samples = enhanced_manifold_mcmc_sampler(self,
            X=X,
            bw=bw,
            num_samples=num_samples,
            k_neighbors=20,
            verbose=False
        )
        return samples
        # return augmented_data.view(-1, self.X.shape[1])

    def sample2(self, num_samples):
        # Compute PDF values for all samples in self.X
        # pdf_values = torch.exp(self.mvn.log_prob(self.X))  # Shape: (n,)
        # # Normalize PDF values to form a probability distribution
        # weights = pdf_values / pdf_values.sum()

        log_pdf_values = self.mvn.log_prob(self.X)  # Shape: (n,)

        # Use log-sum-exp trick for numerical stability
        max_log_val = torch.max(log_pdf_values)
        log_weights = log_pdf_values - max_log_val - torch.log(torch.sum(torch.exp(log_pdf_values - max_log_val)))

        # Convert back to linear space if needed, but often you can work in log space
        pdf_values = torch.exp(log_weights)
        weights = pdf_values / pdf_values.sum()

        # Sample indices based on the weights
        # idxs = torch.multinomial(weights, num_samples, replacement=False)
        # Add noise to the selected samples
        # Sort weights in descending order and get the indices of the top-k
        sorted_indices = torch.argsort(weights, descending=True)
        idxs = sorted_indices[:num_samples]
        if (idxs >= len(self.X)).any():
            idxs = torch.clamp(idxs, 0, len(self.X) - 1)
            print("Warning: clamped indices to valid range")
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    # sample but keep the manifold structure
    def sample4(self, num_samples, K=10, alpha=1.0):
        # 1. 原始PDF权重计算（保留数值稳定性）
        log_pdf_values = self.mvn.log_prob(self.X)
        max_log_val = torch.max(log_pdf_values)
        log_weights = log_pdf_values - max_log_val - torch.log(torch.sum(torch.exp(log_pdf_values - max_log_val)))
        pdf_values = torch.exp(log_weights)  # 归一化PDF值

        # 2. 计算局部密度（K近邻平均距离）
        from torch.nn import PairwiseDistance
        dist = PairwiseDistance(p=2)
        n = len(self.X)
        local_density = torch.zeros(n, device=self.X.device)
        for i in range(n):
            # 计算样本i与所有其他样本的距离
            distances = dist(self.X[i].unsqueeze(0), self.X)
            # 取K近邻距离（排除自身）
            knn_dist = torch.sort(distances)[0][1:K + 1]  # [0]为自身距离0
            local_density[i] = torch.mean(knn_dist)  # 平均距离越大，密度越低

        # 3. 几何调整权重（稀疏区域权重提升）
        adjusted_weights = pdf_values * (local_density ** alpha)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()  # 重新归一化

        # 4. 多项式采样（而非硬Top-K）
        idxs = torch.multinomial(adjusted_weights, num_samples, replacement=False)
        # （后续噪声添加步骤不变）
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def sample(self, num_samples, K=15, alpha=1.0):
        # Ensure K is a scalar integer
        if torch.is_tensor(K):
            K = K.item() if K.numel() == 1 else 10  # Extract scalar or use default

        # 1. Original PDF weight calculation
        log_pdf_values = self.mvn.log_prob(self.X)
        max_log_val = torch.max(log_pdf_values)
        log_weights = log_pdf_values - max_log_val - torch.log(torch.sum(torch.exp(log_pdf_values - max_log_val)))
        pdf_values = torch.exp(log_weights)

        # 2. Efficient local density calculation
        n = len(self.X)

        # Compute pairwise distance matrix
        pairwise_dists = torch.cdist(self.X, self.X, p=2)

        # Set diagonal to large value to exclude self
        pairwise_dists.fill_diagonal_(float('inf'))

        # Get K nearest neighbors for each point
        k_neighbors = min(int(K), n - 1)  # Convert to integer explicitly
        if k_neighbors > 0:
            knn_dists, _ = torch.topk(pairwise_dists, k=k_neighbors, dim=1, largest=False)
            local_density = torch.mean(knn_dists, dim=1)
        else:
            local_density = torch.ones(n, device=self.X.device)

        # 3. Geometric adjusted weights
        adjusted_weights = pdf_values * (local_density ** alpha)
        adjusted_weights = adjusted_weights / (adjusted_weights.sum() + 1e-8)

        # 4. Multinomial sampling
        idxs = torch.multinomial(adjusted_weights, num_samples, replacement=False)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def sample_manifold_aware(self, num_samples, K=10, alpha=1.0, diversity_factor=0.3):
        # Ensure K is a scalar integer
        if torch.is_tensor(K):
            K = K.item() if K.numel() == 1 else 10

        # 1. Original PDF weight calculation
        log_pdf_values = self.mvn.log_prob(self.X)
        max_log_val = torch.max(log_pdf_values)
        log_weights = log_pdf_values - max_log_val - torch.log(torch.sum(torch.exp(log_pdf_values - max_log_val)))
        pdf_values = torch.exp(log_weights)

        # 2. Efficient local density calculation
        n = len(self.X)

        # Compute pairwise distance matrix
        pairwise_dists = torch.cdist(self.X, self.X, p=2)
        pairwise_dists.fill_diagonal_(float('inf'))

        # Get K nearest neighbors for each point
        k_neighbors = min(int(K), n - 1)
        if k_neighbors > 0:
            knn_dists, _ = torch.topk(pairwise_dists, k=k_neighbors, dim=1, largest=False)
            local_density = torch.mean(knn_dists, dim=1)
        else:
            local_density = torch.ones(n, device=self.X.device)

        # 3. NEW: Diversity-aware weight adjustment
        # Combine PDF with diversity consideration
        pdf_weights = pdf_values * (local_density ** alpha)

        # Add explicit diversity term
        if n > 1:
            diversity_weights = self._compute_diversity_weights(pdf_weights, diversity_factor)
            combined_weights = (1 - diversity_factor) * pdf_weights + diversity_factor * diversity_weights
        else:
            combined_weights = pdf_weights

        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)

        # 4. NEW: Two-stage sampling for better coverage
        if num_samples < n:
            # Stage 1: Select diverse high-density samples
            selected_idxs = self._diverse_sampling(combined_weights, self.X, num_samples)
        else:
            # If we need more samples than available, use multinomial
            selected_idxs = torch.multinomial(combined_weights, num_samples, replacement=True)

        # 5. Add noise with adaptive scale based on local density
        noise_scales = self.bw * (local_density[selected_idxs] / local_density.mean())
        norm = Normal(loc=self.X[selected_idxs], scale=noise_scales.unsqueeze(1))
        return norm.sample()

    def _compute_diversity_weights(self, pdf_weights, diversity_factor):
        """Compute weights that favor under-sampled regions"""
        n = len(pdf_weights)

        # Inverse of PDF weights to favor less dense regions
        diversity_weights = 1.0 / (pdf_weights + 1e-8)

        # Smooth the diversity weights
        diversity_weights = torch.log(1 + diversity_weights)

        return diversity_weights / (diversity_weights.sum() + 1e-8)

    def _diverse_sampling(self, weights, points, num_samples):
        """Select samples that are both high-weight and diverse"""
        n = len(weights)

        # Method 1: Determinantal Point Process (DPP) inspired sampling
        selected_idxs = []
        remaining_idxs = torch.arange(n, device=points.device)
        remaining_weights = weights.clone()

        for _ in range(num_samples):
            # Select sample based on current weights
            if len(remaining_idxs) == 0:
                break

            idx = torch.multinomial(remaining_weights, 1).item()
            selected_idx = remaining_idxs[idx]
            selected_idxs.append(selected_idx)

            # Reduce weights of similar points
            selected_point = points[selected_idx]
            similarities = torch.exp(-torch.cdist(selected_point.unsqueeze(0), points).squeeze())
            remaining_weights = remaining_weights * (1 - similarities[remaining_idxs])

            # Remove selected index
            mask = torch.ones(len(remaining_idxs), dtype=torch.bool, device=points.device)
            mask[idx] = False
            remaining_idxs = remaining_idxs[mask]
            remaining_weights = remaining_weights[mask]

            if len(remaining_weights) > 0:
                remaining_weights = remaining_weights / (remaining_weights.sum() + 1e-8)

        return torch.tensor(selected_idxs, device=points.device)


    def dist(self, Y):
        Y_chunks = Y.split(100)
        log_probs = []
        for y in Y_chunks:
            log_prob = self.mvn.log_prob((self.X.unsqueeze(1) - y) / self.bw).double()

            # log_prob = (log_prob - log_prob.min()) * (max_val - min_val) / (
            #         log_prob.max() - log_prob.min()) + min_val

            log_prob = torch.mean(log_prob, dim=0)
            probs = log_prob.detach().cpu().numpy()
            log_probs.extend(probs)

        return log_probs

    def pdf(self, Y):
        max_val = 0
        min_val = -700
        Y = Y.cuda()
        Y_chunks = Y.split(500)
        log_probs = []
        for y in Y_chunks:
            log_prob = self.mvn.log_prob((self.X.unsqueeze(1) - y) / self.bw).double()
            log_prob = (log_prob - log_prob.min()) * (max_val - min_val) / (
                    log_prob.max() - log_prob.min()) + min_val
            probs = torch.mean(log_prob, dim=0)
            probs = probs.detach().cpu().numpy()
            probs = np.exp(probs)
            log_probs.extend(probs)
        return log_probs

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        min_val = -700
        max_val = -1
        if X is None:
            X = self.X
        X = X.cuda()

        log_prob = self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw).double()

        num = int(X.shape[0] * 0.05)
        log_prob, _ = torch.topk(log_prob, k=num, dim=0, sorted=True)
        # log_prob = (log_prob - log_prob.min()) * (max_val - min_val) / (
        #         log_prob.max() - log_prob.min()) + min_val

        # temp = torch.mean(log_prob, dim=1)
        # mask = torch.logical_and(temp >= self.low_train, temp <= self.high_train)
        # log_prob11 = log_prob[mask, :]
        # lower_limit, _ = torch.kthvalue(log_prob, int(0.025 * log_prob.numel()))
        # upper_limit, _ = torch.kthvalue(log_prob, int(0.975 * log_prob.numel()))  # Use torch.clamp to restrict the values of log probabilities to the desired range
        # log_probs = torch.clamp(log_prob, lower_limit.detach().item(), upper_limit.detach().item())

        # log_prob = torch.clamp(log_prob, self.low, self.high)# print(torch.min(log_prob))
        # mask = (log_prob > self.low).all(dim=1)
        # log_probs = log_prob[mask]
        # print(log_prob.shape[0])
        # print(log_probs.shape[0])



        # print(torch.max(log_prob))
        # log_probs = torch.clamp(log_prob, self.low_train, self.high_train)

        # mask = torch.logical_and(log_prob >= self.low_train, log_prob <= self.high_train)
        # log_prob = torch.where(mask, log_prob, torch.zeros_like(log_prob))

        # low = torch.quantile(log_prob, 0.1)
        # if low < -730:
        #     print(low)
        # log_prob = torch.clamp(log_prob, -730, 700)
        # pre = torch.log(self.base)
        # pre = 1e100
        # log_prob = torch.clamp(log_prob / self.temperature, -730, 700)
        # prob = torch.exp(log_prob)
        # m = torch.quantile(prob, 0.5)
        # pre = self.base
        # pre = self.bw ** (-self.dims / 1)
        # sumexp = (pre * prob).sum(dim=0) / self.n
        # sumexp = torch.nan_to_num(sumexp)
        # clamp_sum_exp = torch.clamp(sumexp, self.low, self.high)  # upper bound 150 is the best
        # clamp_sum_exp = torch.nan_to_num(sumexp)
        # clamp_sum_exp = torch.clamp(sumexp, 1e-250, 1e50)
        # log_probs = torch.log(clamp_sum_exp)

        # log_probs = log_prob + pre - torch.log(torch.tensor(self.n))
        # log_probs = torch.mean(log_prob, dim=0)

        # log_prob = torch.mean(log_prob, dim=0)

        max_log_val, _ = torch.max(log_prob, dim=0, keepdim=True)
        relative_log_probs = log_prob - max_log_val
        sum_exp = torch.sum(torch.exp(relative_log_probs), dim=0)
        log_prob = torch.log(sum_exp) + max_log_val.squeeze(0)

        # mask = torch.logical_and(log_prob1 >= self.low_train, log_prob1 <= self.high_train)
        # log_probs = log_prob1[mask]

        # log_probs = torch.log(
        #     (self.bw**(-self.dims) *3]
        #      torch.exp(self.mvn.log_prob(
        #          (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)
        return log_prob

    def score_samples1(self, Y, X=None):
        samplesize = 10
        if X is None:
            X = self.X
        b = Y.shape[0]
        Y = Y.repeat(samplesize, 1, 1)
        Y = torch.randn_like(Y) * 0.0001 + Y
        Y = Y.reshape(-1, X.shape[1])
        log_prob = self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw).double()
        log_prob = log_prob.reshape(X.shape[0], b, samplesize)
        log_prob = torch.quantile(log_prob, 0.05, dim=2)
        log_probs = torch.clamp(log_prob, -800000, 1000)
        log_prob = torch.mean(log_probs, dim=0)
        return log_prob

    def score_samples_test(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        min_val = -700
        max_val = -1

        if X is None:
            X = self.X
        log_prob = self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw).double()

        # log_prob = (log_prob - log_prob.min()) * (max_val - min_val) / (
        #         log_prob.max() - log_prob.min()) + min_val

        # temp = torch.mean(log_prob, dim=1)
        # mask = torch.logical_and(temp >= self.low_train, temp <= self.high_train)
        # log_prob11 = log_prob[mask, :]
        # log_prob = torch.clamp(log_prob, self.low_train , self.high_train)
        log_prob11 = torch.clamp(log_prob, self.low, 0)


        # log_prob = torch.where(mask, log_prob, torch.zeros_like(log_prob))

        # log_prob = torch.clamp(log_prob, self.low, self.high)
        # low = torch.quantile(log_prob, 0.1)
        # if low < -730:
        #     print(low)
        # log_prob = torch.clamp(log_prob, -730, 700)
        # pre = torch.log(self.base)
        # pre = 1e100
        # log_prob = torch.clamp(log_prob / self.temperature, -730, 700)
        # prob = torch.exp(log_prob)
        # m = torch.quantile(prob, 0.5)
        # pre = self.base
        # pre = self.bw ** (-self.dims / 1)
        # sumexp = (pre * prob).sum(dim=0) / self.n
        # sumexp = torch.nan_to_num(sumexp)
        # clamp_sum_exp = torch.clamp(sumexp, self.low, self.high)  # upper bound 150 is the best
        # clamp_sum_exp = torch.nan_to_num(sumexp)
        # clamp_sum_exp = torch.clamp(sumexp, 1e-250, 1e50)
        # log_probs = torch.log(clamp_sum_exp)

        # log_probs = log_prob + pre - torch.log(torch.tensor(self.n))

        log_prob = torch.mean(log_prob11, dim=0)
        # mask = torch.logical_and(log_prob1 >= self.low_train, log_prob1 <= self.high_train)
        # log_probs = log_prob1[mask]


        # log_prob = torch.clamp(log_prob, self.low, self.high)
        # log_probs = torch.log(
        #     (self.bw**(-self.dims) *3]
        #      torch.exp(self.mvn.log_prob(
        #          (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)
        return log_prob

    def log_prob1(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        # X_chunks = self.X.split(1000)
        Y_chunks = Y.split(200)
        log_prob = 0
        try:
            for y in Y_chunks:
                log_prob += self.score_samples(y, self.X).sum(dim=0)
        except:
            return torch.nan
        return log_prob / (Y.shape[0])

    def log_prob_w(self, Y, w):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        # X_chunks = self.X.split(1000)
        # w = torch.tensor(w)
        # log_prob = torch.tensor(0).cuda()
        try:

            log_prob = self.score_samples(Y, self.X) * w
        except:
            return torch.nan
        return torch.sum(log_prob) / torch.sum(w)

    def score_samples5(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y` with improved numerical stability.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.

        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X is None:
            X = self.X
        X = X.to(Y.device)

        # Calculate the squared distances efficiently
        # (X.unsqueeze(1) - Y.unsqueeze(0)) has shape (n, m, d)
        # We want to compute the squared Mahalanobis distance for each pair
        diff = (X.unsqueeze(1) - Y.unsqueeze(0)) / self.bw  # Shape: (n, m, d)

        # For identity covariance, the squared Mahalanobis distance is just the squared Euclidean distance
        squared_distances = 0.5 * torch.sum(diff ** 2, dim=-1)  # Shape: (n, m)

        # Calculate the log normalization constant
        # For a multivariate normal with identity covariance: -0.5 * d * log(2π)
        log_normalization = -0.5 * self.dims * math.log(2 * math.pi)

        # Calculate the log probabilities without the normalization constant
        log_kernel = log_normalization - squared_distances  # Shape: (n, m)

        # Use logsumexp for numerical stability when summing probabilities in log space
        # This avoids underflow when dealing with very small probabilities
        max_log_val, _ = torch.max(log_kernel, dim=0, keepdim=True)
        relative_log_probs = log_kernel - max_log_val
        sum_exp = torch.sum(torch.exp(relative_log_probs), dim=0)
        log_sum_exp = torch.log(sum_exp) + max_log_val.squeeze(0)

        # Apply bandwidth adjustment and normalize by number of samples
        # The bandwidth factor: (1/bandwidth)^d
        bandwidth_factor = -self.dims * torch.log(self.bw)
        log_probs = log_sum_exp + bandwidth_factor - torch.log(
            torch.tensor(X.shape[0], dtype=torch.float, device=Y.device))

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """
        Y_chunks = Y.split(200)
        total_log_prob = 0
        total_points = 0

        try:
            for y_chunk in Y_chunks:
                chunk_log_probs = self.score_samples(y_chunk, self.X)
                total_log_prob += torch.sum(chunk_log_probs)
                total_points += y_chunk.shape[0]

            # Return the average log probability
            return total_log_prob / total_points if total_points > 0 else torch.tensor(float('-inf'))
        except:
            return torch.tensor(float('nan'))
#
# X, y = make_blobs(5000, centers=[[0.1, 0.3], [-0.2, -0.1]], cluster_std=0.1)
# kde = GaussianKDE(X=torch.tensor(X, dtype=torch.double32), bw=0.03)
# test_pts = torch.tensor([[-0.75, -0.25], [0.6, 0.4]], requires_grad=True)
# optimizer = LBFGS([test_pts])
#
# for _ in range(10):  # take 10 optimization steps
#     def closure():
#         optimizer.zero_grad()
#         loss = -kde.log_prob(test_pts)
#         loss.backward()
#         return loss
#     optimizer.step(closure)
#
# print(test_pts)

