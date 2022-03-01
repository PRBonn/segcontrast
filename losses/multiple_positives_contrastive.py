import torch
import numpy as np

class MultiPositivesContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        torch.nn.Module.__init__(self)
        self.temperature = temperature

    # forward pass with both augmented samples projections
    # given two vector projections xi and xj
    def forward(self, feats, ids):
        loss = torch.tensor(0., dtype=torch.float64)

        for i in range(len(feats)):
            for j in torch.where(ids[i] == ids)[0]:
                if i == j:
                    continue

                # cosine similarity between positive pair
                sim_positive = feats[i] @ feats[j]
                sim_positive /= torch.norm(feats[i]) * torch.norm(feats[j])
                sim_positive /= self.temperature
                sim_positive = torch.exp(sim_positive)

                # concatenate all samples (negatives + pair of positives)
                x = torch.cat((feats[i].view(1,-1), feats[j].view(1,-1), feats[ids != ids[i]]))
                is_cuda = x.is_cuda

                # compute similarity between all concatenated samples
                sim_all = torch.mm(x, x.T)
                sim_all /= torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T).clamp(min=1e-16)
                sim_all /= self.temperature
                sim_all = torch.exp(sim_all)

                # subtract diagonals because similarity between x[i],x[i] should not be considered
                diag_sum = torch.exp(torch.ones(x.size(0)) / self.temperature)
                diag_sum = diag_sum.cuda() if is_cuda else diag_sum

                # compute the loss
                loss += torch.mean(
                    -torch.log(
                        sim_positive / (torch.sum(sim_all, dim=-1) - diag_sum)
                    )
                )
        
        return loss

loss = MultiPositivesContrastiveLoss()
feats = torch.from_numpy(np.asarray([[.0,.1,.1,.2],
                                        [.9,.2,.7,.5],
                                        [.1,.3,.2,.2],
                                        [.8,.4,.9,.3]]))
ids = torch.from_numpy(np.asarray([1,2,1,3]))
loss(feats, ids)
