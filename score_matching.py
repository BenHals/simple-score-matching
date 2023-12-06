import torch
import torch.autograd as autograd


def exact_1d_score_matching(scores: torch.Tensor, samples: torch.Tensor):
    """
    The score matching loss takes a model which outputs the log probability
    for a set of samples taken from a distribution. The aim of the loss function
    is to train the model to better match the distribution samples were drawn
    from.

    It is difficult to directly train on the objective, as any given parameterized
    distribution learned by the model needs some normalization constant to be a true
    probability distribution, i.e., to add to one. Predicting a normalized model
    constrains which models we can use, but predicting an unnormalized model makes it
    very difficult to assess the quality of the output (as the model can just put high
    probability over the entire range).

    Instead, we can use the score function, the derivative of the probability distribution,
    as an objective. The idea is that we can optimize an unnormalized model by trying to
    get the derivative it is predicting to match that seen in the data.

    The next problem is then that we do not have the gradient of the data distribution.
    The solution is score matching. The idea is that we instead use the second derivative
    at each of the sample points and optimize this to be zero, i.e., make the samples
    maximum points.

    This optimization has two key parts:
    - The norm of the score function at each of the sample points,
    - The trace of the hessian at each of the sample points

    Unfortunately the hessian is difficult to calculate, as we have to
    to n forward and backwards passes to calculate each element of the
    diagonal. Techniques like sliced score matching speed this up, but in the
    1d case this is fine.
    """

    # Negative log probability of the samples as predicted by
    # the model. Sum across all samples, because this is how
    # log probabilities are aggregated.
    # Assumes that the model predicts log probability.

    # grad1 = autograd.grad(scores, samples, create_graph=True, retain_graph=True)[0]

    # First part of the loss, the norm of the gradient at the samples
    loss1 = torch.norm(scores, dim=-1) ** 2 / 2

    loss2 = torch.zeros(samples.shape[0], device=samples.device)

    # This is the second derivative of loss, i.e., the rate of
    # change of the gradient.
    grad2 = autograd.grad(
        scores[:, 0].sum(), samples, create_graph=True, retain_graph=True
    )[0][:, 0]
    loss2 += grad2

    loss = loss1 + loss2

    return loss


def ssm_loss(model, x, v):
    """SSM loss from
    Sliced Score Matching: A Scalable Approach to Density and Score Estimation

    The loss is computed as
    s = -dE(x)/dx
    loss = vT*(ds/dx)*v + 1/2*(vT*s)^2

    Args:
        x (torch.Tensor): input samples
        v (torch.Tensor): sampled noises

    Returns:
        SSM loss
    """
    x = x.unsqueeze(0).expand(5, *x.shape)  # (n_slices, b, ...)
    x = x.contiguous().view(-1, *x.shape[2:])  # (n_slices*b, ...)
    x = x.requires_grad_()
    score = model.score(x)  # (n_slices*b, ...)
    sv = torch.sum(score * v)  # ()
    loss1 = torch.sum(score * v, dim=-1) ** 2 * 0.5  # (n_slices*b,)
    gsv = torch.autograd.grad(sv, x, create_graph=True)[0]  # (n_slices*b, ...)
    loss2 = torch.sum(v * gsv, dim=-1)  # (n_slices*b,)
    loss = (loss1 + loss2).mean()  # ()
    return loss
