import torch
import tqdm
import torch.nn as nn
import numpy as np

def SinexC_AUDIO_MAE(segment_mat, model, x, alpha, beta):
    """"SinexC algorithm for AUDIO-MAE.
        segment_mat shape: (timesteps, frequency)
        alpha: Number of coalitions per segment
        beta: Fraction of included segments in each coalition
    """
    
    assert len(segment_mat.shape) == 2, "Segment matrix must be two-dimensional"
    assert segment_mat.shape[0] == 1024, "Segment matrix must have 1024 timesteps"
    assert segment_mat.shape[1] == 128, "Segment matrix must have 128 frequency bins"
    assert len(x.shape) == 4, "Input x must be of dimension (1, 1, 1024, 128)"
    assert x.shape[0] == 1, "Input x must be of dimension (1, 1, 1024, 128)"
    assert x.shape[1] == 1, "Input x must be of dimension (1, 1, 1024, 128)"
    assert x.shape[2] == 1024, "Input x must be of dimension (1, 1, 1024, 128)"
    assert x.shape[3] == 128, "Input x must be of dimension (1, 1, 1024, 128)"
    assert beta <= 1, "Fraction of segments per sample must be less than or equal to 1"

    pdist = nn.CosineSimilarity(dim=1)
    t = segment_mat.shape[0]
    f = segment_mat.shape[1]

    R = torch.zeros((t, f), dtype=torch.float32) # Importance
    U = torch.zeros((t, f), dtype=torch.float32) # Uncertainty

    num_segments = np.unique(segment_mat).shape[0]
    min_x = x.min()

    with torch.no_grad():
        h_star = model.forward_encoder_no_mask(x)[:,1:,:].mean(dim=1)

        #Importance
        for i in tqdm.trange(num_segments, desc=f"Processing segments for AUDIO-MAE"):
            segment_mask = (segment_mat == i)
            P = np.random.binomial(1, beta, size=(alpha, num_segments))
            s = 0
            for p in P:
                extra_masks = np.any([segment_mat == j for j in range(num_segments) if p[j] == 1], axis = 0)
                if np.sum(extra_masks) > 0:
                    replace_mask = torch.from_numpy(np.any([segment_mask, extra_masks], axis = 0))
                else:
                    replace_mask = torch.from_numpy(segment_mask)
                #weight_mask = replace_mask.sum()

                x_new = replace_mask * x[0,0,:,:] + min_x * ~replace_mask
                x_new = x_new.unsqueeze(0)
                x_new = x_new.unsqueeze(0)
                h_new = model.forward_encoder_no_mask(x_new)[:,1:,:].mean(dim=1)
                s_new = pdist(h_star, h_new) # / weight_mask
                s += s_new

            R += (s / alpha) * segment_mask
        
        #Uncertainty
        for i in tqdm.trange(num_segments, desc=f"Processing segments for AUDIO-MAE"):
            segment_mask = (segment_mat == i)
            P = np.random.binomial(1, beta, size=(alpha, num_segments))
            var = 0

            for p in P:
                extra_masks = np.any([segment_mat == j for j in range(num_segments) if p[j] == 1], axis = 0)
                if np.sum(extra_masks) > 0:
                    replace_mask = torch.from_numpy(np.any([segment_mask, extra_masks], axis = 0))
                else:
                    replace_mask = torch.from_numpy(segment_mask)
                #weight_mask = replace_mask.sum()

                x_new = replace_mask * x[0,0,:,:] + min_x * ~replace_mask
                x_new = x_new.unsqueeze(0)
                x_new = x_new.unsqueeze(0)
                h_new = model.forward_encoder_no_mask(x_new)[:,1:,:].mean(dim=1)
                s_new = pdist(h_star, h_new) # / weight_mask
                var += (s_new-R)**2
                
            U += (var / (alpha-1)) * segment_mask

    return R, U