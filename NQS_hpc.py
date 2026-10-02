# Optimized, device-agnostic rewrite of NQS.py: Neural-Network Quantum State (RBM ansatz)
# + Variational Monte Carlo + Stochastic Reconfiguration for the 1D transverse-field Ising
# model, with a batched multi-chain MCMC sampler so the whole pipeline can run on GPU.
#
# Wavefunction/Hamiltonian formulas are unchanged from NQS.py; the rewrite targets the
# sampler, the energy-averaging glue code, and device/dtype placement. NQS.py itself is left
# untouched as the legacy/reference implementation.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn


class RBM(nn.Module):
    # Same architecture as NQS.RBM: N visible units (spins), M hidden units, bipartite.

    def __init__(self, num_visible, num_hidden, device=None, dtype=torch.float32, seed=None):
        super().__init__()
        self.N = num_visible
        self.M = num_hidden
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        if seed is not None:
            torch.manual_seed(seed)
        self.W = nn.Parameter(
            torch.randn(num_visible, num_hidden, dtype=dtype) * 0.01
        )
        self.v = nn.Parameter(torch.randn(num_visible, dtype=dtype))
        self.h = nn.Parameter(torch.randn(num_hidden, dtype=dtype))
        self.to(self.device)

    def log_psi(self, s):
        # log(psi(s)) = s . v + sum_j log(cosh(h_j + s . W_:,j))
        theta_j = self.h + s @ self.W
        hidden_term = torch.log(torch.cosh(theta_j))
        linear_term = s @ self.v
        return linear_term + torch.sum(hidden_term, dim=1)

    def psi(self, s):
        return torch.exp(self.log_psi(s))

    def log_derivatives(self, s):
        theta = self.h + s @ self.W
        tanh_theta = torch.tanh(theta)

        O_v = s
        O_h = tanh_theta
        O_W = (s.unsqueeze(2) * tanh_theta.unsqueeze(1)).reshape(s.shape[0], -1)

        return torch.cat([O_v, O_h, O_W], dim=1)


class TFI:
    # Transverse-field Ising Hamiltonian: H = -J sum_i sz_i sz_{i+1} - h sum_i sx_i
    # (periodic boundary conditions). Local-energy formulas unchanged from NQS.TFI.

    def __init__(self, N, J=1.0, h=1.0):
        self.N = N
        self.J = J
        self.h = h

    def diagonal_energy(self, s):
        interaction = s * torch.roll(s, shifts=-1, dims=1)
        return interaction.sum(dim=1)

    def _flip_spin(self, s, i):
        s_flipped = s.clone()
        s_flipped[:, i] *= -1
        return s_flipped

    def off_diagonal_energy(self, s, wavefunction):
        ratios = torch.stack([
            wavefunction.psi(self._flip_spin(s, i)) / wavefunction.psi(s)
            for i in range(self.N)
        ])
        return ratios.sum(dim=0)

    def loc_energy(self, s, wavefunction):
        return -self.J * self.diagonal_energy(s) - self.h * self.off_diagonal_energy(s, wavefunction)


class BatchedMCMC:
    # Runs n_chains independent Metropolis chains simultaneously, each a single-spin-flip
    # sampler targeting |psi(s)|^2, like NQS.MCMC — but every step is one batched tensor op
    # across all chains instead of a Python loop over individual chains/samples. This is what
    # lets GPU actually help, and is the main bottleneck fix relative to NQS.py.

    def __init__(self, n_spins, n_chains, device=None, dtype=torch.float32, seed=None):
        self.n_spins = n_spins
        self.n_chains = n_chains
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        if seed is not None:
            torch.manual_seed(seed)
        self.s = self.random_state()

    def random_state(self):
        return (
            2 * torch.randint(0, 2, (self.n_chains, self.n_spins), device=self.device) - 1
        ).to(self.dtype)

    @torch.no_grad()
    def step(self, wavefunction):
        # One random site per chain
        idx = torch.randint(self.n_spins, (self.n_chains,), device=self.device)

        proposed = self.s.clone()
        proposed[torch.arange(self.n_chains, device=self.device), idx] *= -1

        log_psi_current = wavefunction.log_psi(self.s)
        log_psi_proposed = wavefunction.log_psi(proposed)

        log_ratio = 2 * (log_psi_proposed - log_psi_current)
        accept = torch.log(torch.rand(self.n_chains, device=self.device)) < log_ratio

        self.s = torch.where(accept.unsqueeze(1), proposed, self.s)

    @torch.no_grad()
    def sample(self, wavefunction, n_steps, burn_in=100):
        # Returns (n_steps * n_chains, n_spins): one sample per chain per recorded step.
        for _ in range(burn_in):
            self.step(wavefunction)

        samples = []
        for _ in range(n_steps):
            self.step(wavefunction)
            samples.append(self.s.clone())

        return torch.cat(samples, dim=0)


class VMC:
    def __init__(self, wavefunction, hamiltonian, mcmc_sampler):
        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.mcmc_sampler = mcmc_sampler

    def local_energy(self, s):
        return self.hamiltonian.loc_energy(s, self.wavefunction)

    def average_local_energy(self, n_steps, burn_in=100):
        samples = self.mcmc_sampler.sample(self.wavefunction, n_steps, burn_in)
        return self.local_energy(samples).mean()


class SR:
    # Stochastic Reconfiguration. Same formulas as NQS.SR; adds an optional matrix-free CG
    # solve so (S + lambda I) can be applied via matvecs instead of forming/factorizing the
    # full P x P Fisher matrix once P = N + M + N*M gets large.

    def __init__(self, learning_rate=0.01, lambda_reg=1e-3, noise_std=0.0, solver="dense",
                 cg_tol=1e-5, cg_maxiter=100):
        # cg_maxiter/cg_tol trade solve accuracy for iterations: CG only pays off over the
        # dense solve once P = N + M + N*M is large enough that O(P^3) factorization
        # dominates O(cg_maxiter * P^2) matvecs -- at small/medium P, a looser tolerance
        # (fewer iterations) is usually the better trade than chasing dense-solve accuracy.
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.noise_std = noise_std
        assert solver in ("dense", "cg")
        self.solver = solver
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

    def FS_metric(self, O):
        O_mean = O.mean(dim=0)
        O_centered = O - O_mean
        return (O_centered.conj().T @ O_centered) / O.shape[0]

    def compute_gradient(self, O, local_energy):
        O_mean = O.mean(dim=0)
        E_mean = local_energy.mean()
        return 2 * ((O * local_energy.unsqueeze(1)).mean(dim=0) - O_mean * E_mean)

    def _solve_dense(self, O, rhs):
        S = self.FS_metric(O)
        I = torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
        S_reg = S + self.lambda_reg * I
        return torch.linalg.solve(S_reg, rhs)

    def _solve_cg(self, O, rhs):
        # (S + lambda I) x = rhs, applying S via matvecs: S x = O^T (O x) / n_samples
        # (mean-centering drops out of S @ x for a constant shift since O_centered @ x
        # and O @ x differ only by a term proportional to O_mean . x, which FS_metric's
        # O_centered already removes on the left via subtracting <O_i><O_j>; we replicate
        # that by centering O once up front).
        n = O.shape[0]
        O_mean = O.mean(dim=0)
        O_centered = O - O_mean

        def matvec(x):
            return (O_centered.conj().T @ (O_centered @ x)) / n + self.lambda_reg * x

        x = torch.zeros_like(rhs)
        r = rhs - matvec(x)
        p = r.clone()
        rs_old = torch.dot(r, r)
        for _ in range(self.cg_maxiter):
            Ap = matvec(p)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < self.cg_tol:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def update(self, wavefunction, samples, local_energy):
        O = wavefunction.log_derivatives(samples)
        gradient = self.compute_gradient(O, local_energy)
        rhs = -self.learning_rate * gradient

        if self.solver == "dense":
            delta_theta = self._solve_dense(O, rhs)
        else:
            delta_theta = self._solve_cg(O, rhs)

        if self.noise_std > 0:
            delta_theta = delta_theta + torch.randn_like(delta_theta) * self.noise_std

        N = wavefunction.N
        M = wavefunction.M
        with torch.no_grad():
            wavefunction.v += delta_theta[:N]
            wavefunction.h += delta_theta[N:N + M]
            wavefunction.W += delta_theta[N + M:].reshape(N, M)


def ed_ground(N, J, h):
    # Exact ground-state energy via sparse diagonalization. Benchmark only: intended for
    # small N (roughly N <= 20-24 depending on available memory) to validate the optimized
    # sampler/SR against a known answer -- not meant to scale to the large-N runs this
    # library targets, which have no exact check at all.
    dim = 2 ** N
    basis = [tuple(1 if (idx >> bit) & 1 == 0 else -1 for bit in range(N)) for idx in range(dim)]
    index = {s: i for i, s in enumerate(basis)}

    diag = np.empty(dim)
    rows, cols, vals = [], [], []
    for j, s in enumerate(basis):
        diag[j] = -J * sum(s[i] * s[(i + 1) % N] for i in range(N))
        for i in range(N):
            flipped = list(s)
            flipped[i] *= -1
            k = index[tuple(flipped)]
            rows.append(k)
            cols.append(j)
            vals.append(-h)

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)) + sp.diags(diag)
    H = H.tocsr()

    if dim <= 4:
        return float(np.linalg.eigvalsh(H.toarray())[0])
    return float(spla.eigsh(H, k=1, which="SA", return_eigenvectors=False)[0])


def run_training(N, M, J=1.0, h=1.0, seed=0, n_chains=64, n_steps=200, burn_in=100,
                  n_iterations=50, device=None, dtype=torch.float32, solver="dense",
                  lr=0.01, lambda_reg=1e-3, noise_std=0.0):
    # One end-to-end training run, used directly by the notebooks (both the small-scale
    # non-HPC one and the large-N HPC one) and by any per-seed/per-ratio sweep loop.
    device = torch.device(device) if device is not None else torch.device("cpu")

    rbm = RBM(num_visible=N, num_hidden=M, device=device, dtype=dtype, seed=seed)
    tfi = TFI(N=N, J=J, h=h)
    mcmc = BatchedMCMC(n_spins=N, n_chains=n_chains, device=device, dtype=dtype, seed=seed + 1_000_000)
    vmc = VMC(rbm, tfi, mcmc)
    sr = SR(learning_rate=lr, lambda_reg=lambda_reg, noise_std=noise_std, solver=solver)

    energy_history = []
    for _ in range(n_iterations):
        samples = mcmc.sample(rbm, n_steps=n_steps, burn_in=burn_in)
        local_energies = vmc.local_energy(samples)
        sr.update(rbm, samples, local_energies)
        energy_history.append(local_energies.mean().item())

    return rbm, energy_history
