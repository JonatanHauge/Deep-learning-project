from Masks import MaskGenerator
import torch.nn as nn
import torch
import tqdm


def RELAX(x, model, mask_bs=1, num_batches=1, mask_percentage=50, mask_type = 'time+frequency', freq_patch = 8, time_patch = 32):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Similarity function
    pdist = nn.CosineSimilarity(dim=1)

    min_x = x.min().to(device)
    input_shape = (x.shape[2], x.shape[3])

    with torch.no_grad():
        h_star = model.forward_encoder_no_mask(x.to(device))[:,0,:] # [1, 768]

    h_star = h_star.expand(mask_bs, -1)

    R = torch.zeros(input_shape, dtype=torch.float32) # Importance
    U = torch.zeros(input_shape) # Uncertainty

    # Loop over batches
    contribution_sum = 0
    for batch_idx in tqdm.trange(num_batches, desc=f"Processing batches (importance)"):
        mask_gen = MaskGenerator(input_shape, mask_type=mask_type, mask_percentage=mask_percentage, mask_bs=mask_bs,
                                 freq_patch = freq_patch, time_patch = time_patch)

        batch_masks = mask_gen.generate().to(device)
        with torch.no_grad():
            replace_mask = (batch_masks == 0)
            x_mask = x * batch_masks + min_x * replace_mask

            x_mask = x_mask.squeeze(0)
            x_mask = x_mask.unsqueeze(1)

            h = model.forward_encoder_no_mask(x_mask.to(device))[:,0,:]   # [mask_bs, 768]
            s = pdist(h_star, h).to('cpu')

            # Update R with the mean over the batch
            contribution_sum += torch.sum(batch_masks, dim = (0,1,2))
            tmp_sum = sum([(s[i] * batch_masks[i].to('cpu')) for i in range(mask_bs)]) / mask_bs
            R = R.add(tmp_sum)

    keep_factor = contribution_sum / (1024*128*mask_bs*num_batches)
    R /= (num_batches * keep_factor.to('cpu'))

    #Uncertainty:
    contribution_sum = 0
    for batch_idx in tqdm.trange(num_batches, desc=f"Processing batches (uncertainty)"):
        mask_gen = MaskGenerator(input_shape, mask_type=mask_type, mask_percentage=mask_percentage, mask_bs=mask_bs, 
                                 freq_patch = freq_patch, time_patch = time_patch)

        batch_masks = mask_gen.generate().to(device)
        with torch.no_grad():

            replace_mask = (batch_masks == 0)
            x_mask = x * batch_masks + min_x * replace_mask
            x_mask = x_mask.squeeze(0)
            x_mask = x_mask.unsqueeze(1)

            h = model.forward_encoder_no_mask(x_mask.to(device))[:,0,:]   # [mask_bs, 768]
            s = pdist(h_star, h).to('cpu')

            # Update U with the mean over the batch
            vars = [(s[i]-R.to('cpu'))**2 for i in range(len(s))]
            contribution_sum += torch.sum(batch_masks, dim = (0,1,2))
            tmp_sum = sum([(vars[i] * batch_masks[i].to('cpu')) for i in range(mask_bs)]) / mask_bs

            U = U.add(tmp_sum)

    keep_factor_factor = contribution_sum / (1024*128*mask_bs*num_batches)
    U /= ((num_batches-1)*keep_factor.to('cpu'))

    return R, U