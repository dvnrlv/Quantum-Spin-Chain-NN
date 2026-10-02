# this is the OOP implementation of the Neural Quantum State (NQS), used to represent and manipulate quantum spin chain states in a neural network framework. 

# I guess I have to pay attention to dim = 1 in torch sums

import numpy as np
import torch
import torch.nn as nn



class RBM(nn.Module): 

    # the RBM class handles the representation of the quantum wave function as parametrised by an RBM function; M hidden and N visible units, bipartite structure.


    def __init__(self, num_visible, num_hidden, k = 1, device = None, seed = None):
        super().__init__()
        self.N = num_visible
        self.M = num_hidden
        if seed is not None:
            torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        self.v = nn.Parameter(torch.randn(num_visible))
        self.h = nn.Parameter(torch.randn(num_hidden))


    def log_psi(self, s):
        # log(phi(s)) = sum_i s_i * v_bias_i + sum_j log(1 + exp(h_bias_j + sum_i W_ij * s_i))
        # s is a tensor representing the spin chain configuration, with shape (batch_size, num_visible)

        theta_j = self.h + s @ self.W  # shape: (batch_size, num_hidden);effective angle for hidden units

        hidden_term = torch.log(torch.cosh(theta_j))

        linear_term = s @ self.v
     
        return linear_term + torch.sum(hidden_term, dim = 1)

    def psi(self, s):
        # psi(s) = exp(logpsi(s))
        return torch.exp(self.log_psi(s))


    def log_derivatives(self, s): 

        theta = self.h + s @ self.W

        tanh_theta = torch.tanh(theta)

        # d log(psi) / d v_i
        O_v = s

        # d log(psi) / d h_j
        O_h = tanh_theta

        # d log(psi) / d W_ij
        O_W = s.unsqueeze(2) * tanh_theta.unsqueeze(1)

        # Flatten W derivatives
        O_W = O_W.reshape(s.shape[0], -1)

        # Combine all derivatives into one vector
        O = torch.cat([O_v, O_h, O_W], dim=1)

        return O




class TFI: # this handles the transverse field Ising model Hamiltonian to compute local energy
    def __init__(self, N, J=1.0, h=1.0):
        self.N = N  # number of spins
        self.J = J  # coupling constant
        self.h = h  # transverse field strength


    def diagonal_energy(self, s): # this represents the interaction term of the Hamiltonian
        """
        Σ_i σ_i σ_{i+1}
        """
        interaction = s * torch.roll(
            s,
            shifts=-1,
            dims=1
        )
        return interaction.sum(dim=1)

    def _flip_spin(self, s, i):
                s_flipped = s.clone()
                s_flipped[:, i] *= -1
                return s_flipped

    def off_diagonal_energy(self, s, wavefunction):
        """
        Σ_i σ_i^x
        """

        # note: σ_i^x flips the spin at site i; this introduces off-diagonal elements andf S is no longer an eigenvalue; hence, the MCMC sampling estimation with ratios.

        """
        E_loc(S) = <S|H|psi> / psi(S) = E_Ising(S) - h * sum_i psi(S^(i)) / psi(S)
        """

        ratios = torch.stack([
            wavefunction.psi(self._flip_spin(s, i)) / wavefunction.psi(s)
            for i in range(self.N)
        ])

        return ratios.sum(dim=0)

    def loc_energy(self, s, wavefunction):

        return - self.J* self.diagonal_energy(s)  - self.h* self.off_diagonal_energy(s, wavefunction)



class MCMC: # this class handles the Markov Chain Monte Carlo (Metropolis) sampling of the spin configurations according to the probability distribution defined by the NQS wave function; prop to |psi(S)|^2, which is achieved by proposing new configurations and accepting/rejecting them based on the acceptance ratio.

# burn-in period is implemented to ensure that the initial sample does not bias the distrubition

    def __init__(self, n_spins, seed=None):
        self.n_spins = n_spins
        if seed is not None:
            torch.manual_seed(seed)
        self.s = self.random_state()

    def random_state(self):
        return (2 * torch.randint(0, 2, (self.n_spins,)) - 1).float()

    @torch.no_grad()
    def step(self, wavefunction):

        # Choose a random spin
        i = torch.randint(self.n_spins, ())

        # Propose a new state by flipping one spin
        proposed_state = self.s.clone()
        proposed_state[i] *= -1

        # wavefunction
        log_psi_current = wavefunction.log_psi(
            self.s.unsqueeze(0)
        )[0]

        log_psi_proposed = wavefunction.log_psi(
            proposed_state.unsqueeze(0)
        )[0]

        # Metropolis ratio
        log_ratio = 2 * (
            log_psi_proposed - log_psi_current
        )

        # accept/rejection
        if torch.log(torch.rand(())) < log_ratio:
            self.s = proposed_state

    @torch.no_grad()
    def sample(self, wavefunction, n_samples, burn_in=100):

        # Burn-in
        for _ in range(burn_in):
            self.step(wavefunction)

        samples = []

        for _ in range(n_samples):
            self.step(wavefunction)
            samples.append(self.s.clone())

        return torch.stack(samples)




class VMC: # here we handle averaging out the local energy over sampled configurations

    def __init__(self, wavefunction, hamiltonian, mcmc_sampler):
        self.wavefunction = wavefunction
        self.hamiltonian = hamiltonian
        self.mcmc_sampler = mcmc_sampler

    def local_energy(self, s):
        return self.hamiltonian.loc_energy(s, self.wavefunction)

    def average_local_energy(self, n_samples, burn_in=100):
        samples = self.mcmc_sampler.sample(
            self.wavefunction,
            n_samples,
            burn_in
        )
        local_energies = torch.tensor([
            self.local_energy(s.unsqueeze(0)) for s in samples
        ])
        return local_energies.mean()


class SR:  # handles parameter backpropogation via Stochastic Reconfiguration, while accounting for Hilbert space by Fubuni-Study metric tensor.
    # note that we know the derivatives of the wavefunction with respect to the parameters, but we need to account for the fact that the wavefunction is a complex function and the parameters are real.
    # Note also the reconfiguration of the S tensor for computational efficiency

    def __init__(self, learning_rate=0.01, lambda_reg=1e-3, noise_std=0.0):

        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.noise_std = noise_std


    def FS_metric(self, O):

        # Mean of the log derivatives
        O_mean = O.mean(dim=0)

        # Centre the log derivatives
        O_centered = O - O_mean

        # S_ij = <O_i O_j> - <O_i><O_j>
        S = (O_centered.conj().T @ O_centered) / O.shape[0]

        # conjugate accounts for complex nature of the wavefunction ()

        return S


    def compute_gradient(self, O, local_energy):

        # Mean values
        O_mean = O.mean(dim=0)
        E_mean = local_energy.mean()

        # g_i = 2 [<O_i E_loc> - <O_i><E_loc>]
        gradient = 2 * (
            (O * local_energy.unsqueeze(1)).mean(dim=0)
            - O_mean * E_mean
        )

        return gradient


    def update(self, wavefunction, samples, local_energy):

        # Calculate log derivatives
        O = wavefunction.log_derivatives(samples)

        # Calculate gradient
        gradient = self.compute_gradient(
            O,
            local_energy
        )

        # Calculate SR metric
        S = self.FS_metric(O)

        # Regularisation: S -> S + lambda I
        I = torch.eye(
            S.shape[0],
            dtype=S.dtype,
            device=S.device
        )

        S_reg = S + self.lambda_reg * I

        # Solve:
        #
        # (S + lambda I) delta_theta = -eta * gradient
        #
        delta_theta = torch.linalg.solve(
            S_reg,
            -self.learning_rate * gradient
        )

        # Optional exploration noise: delta_theta = -eta * S_lambda^-1 * g + epsilon, epsilon_i ~ N(0, sigma^2)
        if self.noise_std > 0:
            epsilon = torch.randn_like(delta_theta) * self.noise_std
            delta_theta = delta_theta + epsilon

        # Update RBM parameters
        N = wavefunction.N
        M = wavefunction.M

        # delta_theta is a 1D tensor of length N + M + N*M, where the first N elements correspond to the visible biases, the next M elements correspond to the hidden biases, and the remaining N*M elements correspond to the weights.

        with torch.no_grad():

            # v
            wavefunction.v += delta_theta[:N]

            # h
            wavefunction.h += delta_theta[N:N+M]

            # W
            wavefunction.W += delta_theta[N+M:].reshape(N, M)

        return 




