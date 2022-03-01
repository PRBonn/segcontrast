import torch

class ContrastiveLoss(torch.nn.Module):
    # from paper the optimal temperature is 0.5
    def __init__(self, temperature=0.5):
        torch.nn.Module.__init__(self)
        self.temperature = temperature

    # forward pass with both augmented samples projections
    # given two vector projections xi and xj
    def forward(self, xi, xj):
        # matching pairs exp(sim(i,j)/t) calculation
        # numerator

        # this way we multiply the vectors on position i = j (diagonals)
        # only positive pairs
        sim_positive = torch.sum(xi * xj, dim=-1)
        sim_positive /= torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
        sim_positive /= self.temperature
        sim_positive = torch.exp(sim_positive)

        # we double the size of positive pairs because we calculate it
        # for both augmented pairs xi and xj
        sim_positive = torch.cat((sim_positive, sim_positive), dim=0)

        # negative pairs exp(sim(i,k)/t) calculation
        # denominator

        # join all augmented images in one vector (2 * batch_size)
        x = torch.cat((xi, xj), dim=0)
        is_cuda = x.is_cuda

        # get a matrix with the similarity between all ik pairs
        # diagonals will be i=k, i.e. similarity = 1
        sim_all = torch.mm(x, x.T)
        sim_all /= torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T).clamp(min=1e-16)
        sim_all /= self.temperature
        sim_all = torch.exp(sim_all)

        # on diagonals we will have the position i=k
        # with similarity value = 1 since the similarity
        # of xi and xi will be 1, so we need to subtract them.
        # To subtract this noise we get a vector with
        # the same size as the diagonals, divide it by
        # temperate t and calculate the exp over it.
        # With this calculation we will have the exact value to
        # subtract to remove the diagonals "noisy" summed values
        diag_sum = torch.exp(torch.ones(x.size(0)) / self.temperature)
        diag_sum = diag_sum.cuda() if is_cuda else diag_sum

        loss = torch.mean(
            -torch.log(
                sim_positive / (torch.sum(sim_all, dim=-1) - diag_sum)
            )
        )
        
        return loss
