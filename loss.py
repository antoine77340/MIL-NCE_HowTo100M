import torch as th


class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)


class PMILNCELoss(th.nn.Module):
    def __init__(self):
        super(PMILNCELoss, self).__init__()

    def code_compare(self, code_a, code_b):
        code_len = code_a.shape[1]//2
        code_a = th.stack([code_a]*code_b.shape[0], dim=1)
        code_b = th.stack([code_b]*code_a.shape[0], dim=0)

        means_a = code_a[:, :, :code_len]
        std_a   = code_a[:, :, code_len:]
        means_b = code_b[:, :, :code_len]
        std_b   = code_b[:, :, code_len:]

        loss = (means_a - means_b)**2 / (std_a**2 + std_b**2) + th.log(std_a**2 + std_b**2)
        loss = 0.5 * (loss.sum(dim=2) + code_len * th.log(2 * 3.1415927410125732))

        return loss

    def forward(self, video_embd, text_embd):
        x = self.code_compare(video_embd, text_embd)
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)
