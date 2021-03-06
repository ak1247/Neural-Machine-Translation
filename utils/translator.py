import torch
import torch.nn as nn
import torch.nn.functional as F
## My functions
from utils.models import Transformer,get_pad_mask,get_subseq_mask

class Translator(nn.Module):
    def __init__(self,model,args):
        super(Translator,self).__init__()

        self.alpha = 0.7
        self.beam_size = args.BEAM_SIZE
        self.max_seq_len = args.MAX_SQN_LEN
        self.src_pad_idx = args.src_pad_idx
        self.trg_pad_idx = args.trg_pad_idx
        self.trg_bos_idx = args.trg_bos_idx
        self.trg_eos_idx = args.trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[self.trg_bos_idx]]))
        self.register_buffer('blank_seqs', 
            torch.full((self.beam_size, self.max_seq_len), self.trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer('len_map', 
            torch.arange(1, self.max_seq_len + 1, dtype=torch.long).unsqueeze(0))

    def _model_decode(self,trg_seq,enc_output,src_mask):
        trg_mask = get_subseq_mask(trg_seq)
        dec_output = self.model.decoder(trg_seq,trg_mask,enc_output,src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output),dim=-1)


    def _get_init_state(self,src_seq,src_mask):
        beam_size = self.beam_size
         
        enc_output = self.model.encoder(src_seq,src_mask)
        dec_output = self._model_decode(self.init_seq,enc_output,src_mask)
        #print(dec_output.shape)
        
        best_k_probs, best_k_idx = dec_output[:,-1,:].topk(beam_size)
        #print(best_k_idx)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        #print(gen_seq)
        gen_seq[:,1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size,1,1)

        return enc_output,gen_seq,scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self,src_seq):
        # assert src_seq.size(0) == 1

        # with torch.no_grad():
        #     src_mask = get_pad_mask(src_seq,self.src_pad_idx)
        #     enc_output,gen_seq,scores = self._get_init_state(src_seq,src_mask)

        #     ans_idx = 0
        #     print(self.max_seq_len)
        #     for i in range(2,self.max_seq_len):
        #         dec_output = self._model_decode(gen_seq[:,:i],enc_output,src_mask)
        #         gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, i)
        #         eos_locs = gen_seq == self.trg_eos_idx 
        #         seq_lens, _ = self.len_map.masked_fill(~eos_locs,self.max_seq_len).min(1)
        #         # print(eos_locs)
        #         # if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:
        #         #     _, ans_idx = scores.div(seq_lens.float() ** self.alpha).max(0)
        #         #     ans_idx = ans_idx.item()
        #         #     break
        #         print(i,scores, gen_seq[ans_idx][:seq_lens[ans_idx]].tolist())
        # return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)
            #print(gen_seq,scores,self.trg_bos_idx,trg_eos_idx)
            #check

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    #print(step,scores, gen_seq[ans_idx][:seq_lens[ans_idx]].tolist())
                    break
                #print(step,scores, gen_seq,eos_locs.sum(dim=-1))
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()


