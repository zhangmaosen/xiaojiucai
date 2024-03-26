from model.model import denoise_net
import torch
import torch.nn as nn

class jiucai_net(denoise_net):
    def __init__(self, args):
        super().__init__(args)
        if args.use_attn :
            self.attn = nn.MultiheadAttention(args.head_dim, args.num_heads)
            self.kvproj = nn.Linear(args.embedding_dimension, 2 * args.head_dim)
            self.qproj = nn.Linear(args.query_embedding_dimension, args.head_dim)
    def forward(self, past_time_feat, mark, future_time_feat, t, text_emb):
        params = list(self.kvproj.named_parameters())
        print(params[0])
        print(params[1])
        input = self.embedding(past_time_feat, mark)
        print(f'input is {input}')
        kv = self.kvproj(input)
        
        print(f'kv is {kv}')
        k, v = kv.chunk(2, dim=-1)
        q = self.qproj(text_emb)
        print('attention doing')
        attn_input = self.attn(q, k, v)
        # Output the distribution of the generative results, the sampled generative results and the total correlations of the generative model.
        output, y_noisy = self.diffusion_gen(attn_input, future_time_feat, t)
        
        # Score matching.
        sigmas_t = self.extract(self.sigmas.to(y_noisy.device), t, y_noisy.shape)
        y = future_time_feat.unsqueeze(1).float()
        y_noisy1 = output.sample().float().requires_grad_()
        E = self.score_net(y_noisy1).sum()

        # The Loss of multiscale score matching.
        grad_x = torch.autograd.grad(E, y_noisy1, create_graph=True)[0]
        dsm_loss = torch.mean(torch.sum(((y-y_noisy1.detach())+grad_x*1)**2*sigmas_t, [1,2,3])).float()
        return output, y_noisy, dsm_loss