"""Tree sampling"""

import math

from torch import Tensor, cat, log, repeat_interleave # pylint: disable=no-name-in-module


def _cprobs_to_cdist(c_probs: Tensor, num_joint_bits: int, idx_layer: int, ndim_cprob_b: int, ndim_cprob_t: int) -> Tensor:
    """Convert conditional probabilities into conditional probability distributions with inteleave for joint probability distribution.

    Args:
        c_probs :: (B, T, Cond=2**(k-1)) - B(L_k=1|L_<k=cond) conditional probabilities of all conditions of layer k
        num_joint_bits                   - The number of joint distribution's bits (q)
        idx_layer                        - Index number 'k' of the `c_probs` layer
        ndim_cprob_b                     - The size of c_prob's batch dimension (batch size)
        ndim_cprob_t                     - The size of c_prob's time dimension (time length)

    Returns:
        :: (B, T, Dist=2**q) - Interleaved B(L_k=1|L_<k=cond) conditional probabilities distribution of all conditions of layer k
    """

    # Conditional Prob.s to Conditional Prob. distributions
    #   (B, T, Cond=2**(k-1)) -> (B, T, Cond=2**(k-1), 1) -> (B, T_s, Cond=2**(k-1), Dist=2)
    c_probs = c_probs.unsqueeze(3)
    #                    B(Lk=0|.), B(Lk=1|.)
    c_dists = cat((1.0-c_probs, c_probs), dim=3)

    # Interleave for joint distribution
    # x2 conditions per bit layer (2**(q-k))
    num_repeat = 2 ** (num_joint_bits - idx_layer)
    ## (B, T, Cond=2**(k-1), Dist=2) -> (B, T, cDist=2**k)
    c_dists = c_dists.reshape((ndim_cprob_b, ndim_cprob_t, -1))
    ## (B, T, cDist=2**k) -> (B, T, cDist=2**q)
    c_dists_interleaved = repeat_interleave(c_dists, num_repeat, dim=2)

    return c_dists_interleaved


def tree_to_pdf(cprobs: Tensor) -> Tensor:
    """Convert Hierarchical conditional bit probabilities to the joint probability.

         B(Lk=1|L<k)     B(Lk=0|L<k=Bs<k)/B(Lk=1|L<k=Bs<k)      P(level)
    L1        0.6                     0.4/0.6               0.4  0.6  0.4  0.6
            /     ╲                                          x    x    x    x
    L2    0.1     0.7      =>    0.9/0.1   0.3/0.7    =>    0.9  0.1  0.3  0.7
         .   .   .   .
    P(s).36/.04/.18/.42                                     0.36 0.04 0.18 0.42

    Args:
        cprobs         :: (B, T, Prob=2**q) - Hierarchical conditional bit probabilities
    Returns
                       :: (B, T, Dist=2**q) - Joint probability distribution of all bits == PD of level
    """

    ndim_b = cprobs.shape[0]
    ndim_t = cprobs.shape[1]
    num_joint_bits = round(math.log2(cprobs.shape[2]))

    # Joint Probability Distribution == Π P(L_k|L_<k)
    prob_dist =                 _cprobs_to_cdist(cprobs[:,:, 2**(1-1): 2**(1)],                num_joint_bits,         1, ndim_b, ndim_t)
    for idx_layer in range(2, num_joint_bits + 1):
        prob_dist = prob_dist * _cprobs_to_cdist(cprobs[:,:, 2**(idx_layer-1) : 2**idx_layer], num_joint_bits, idx_layer, ndim_b, ndim_t)
    return prob_dist


def tree_to_logpdf(cprobs: Tensor) -> Tensor:
    """Convert Hierarchical conditional bit probabilities to the log joint probability.

         B(Lk=1|L<k)     B(Lk=0|L<k=Bs<k)/B(Lk=1|L<k=Bs<k)      P(level)
    L1        0.6                     0.4/0.6               0.4  0.6  0.4  0.6
            /     ╲                                          x    x    x    x
    L2    0.1     0.7      =>    0.9/0.1   0.3/0.7    =>    0.9  0.1  0.3  0.7
         .   .   .   .
    P(s).36/.04/.18/.42                                     0.36 0.04 0.18 0.42

    Args:
        cprobs         :: (B, T, Prob=2**q) - Hierarchical conditional bit probabilities
    Returns
                       :: (B, T, Dist=2**q) - Joint probability distribution of all bits == PD of level
    """

    ndim_b = cprobs.shape[0]
    ndim_t = cprobs.shape[1]
    num_joint_bits = round(math.log2(cprobs.shape[2]))

    # Log of Joint Probability Distribution == log(Π P(L_k|L_<k)) == ΣlogP(L_k|L_<k)
    cprob_sum =                 log(_cprobs_to_cdist(cprobs[:,:, 2**(1-1)         : 2**(1)      ], num_joint_bits,         1, ndim_b, ndim_t))
    for idx_layer in range(2, num_joint_bits + 1):
        cprob_sum = cprob_sum + log(_cprobs_to_cdist(cprobs[:,:, 2**(idx_layer-1) : 2**idx_layer], num_joint_bits, idx_layer, ndim_b, ndim_t))
    return cprob_sum
