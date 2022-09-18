import torch

B=32
beam_size=10
V=6225

def compute_topk(scores):
    topk_res = torch.topk(scores.reshape(B, beam_size*V), k=beam_size, dim=-1) # [B, beam_size] [64,10]*2,一个概率，一个ids

    exploratory_accum_scores = topk_res[0]#[64,10]
    exploratory_ids = topk_res[1] % V #[64,10]
    exploratory_beam_from = (topk_res[1] / V).long()

    return exploratory_beam_from

# scores_cpu = torch.load("./inpt_scores.pt")
# torch.save(scores_cpu, "../top_debug_verify/inpt_scores_torch14.pt", _use_new_zipfile_serialization=False)

# scores_cpu = torch.load("../top_debug_verify/inpt_scores_torch14.pt").cpu()
# scores_cuda  = torch.load("../top_debug_verify/inpt_scores_torch14.pt").cuda()

scores_cpu = torch.load("./inpt_scores.pt").cpu()
scores_cuda  = torch.load("./inpt_scores.pt").cuda()

exploratory_beam_from_cpu = compute_topk(scores_cpu)
exploratory_beam_from_cuda = compute_topk(scores_cuda)
# print(exploratory_beam_from_cpu, exploratory_beam_from_cpu.shape)
print("mean diff value is: ", torch.mean(torch.abs(exploratory_beam_from_cpu.cuda() - exploratory_beam_from_cuda).float()))
