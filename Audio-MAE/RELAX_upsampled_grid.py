from Masks import MaskGenerator
import torch.nn as nn
import torch
import tqdm


def RELAX_upsampled_grid(x, model, mask_bs=1, num_batches=1, mask_percentage=50):


    input_shape = (x.shape[2], x.shape[3])
    # Similarity function
    pdist = nn.CosineSimilarity(dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    min_x = x.min().to(device)
    diff_mat = (x.squeeze(0).squeeze(0) - min_x).to(device)

    with torch.no_grad():
        h_star = model.forward_encoder_no_mask(x.to(device))[:,0,:] # [1, 768]

    h_star = h_star.expand(mask_bs, -1)
    R = torch.zeros(input_shape, dtype=torch.float32) # Importance
    U = torch.zeros(input_shape) # Uncertainty

    # Loop over batches
    contribution_sum = 0
    for batch_idx in tqdm.trange(num_batches, desc=f"Processing batches (importance)"):
        mask_gen = MaskGenerator(input_shape, mask_type='RELAX', mask_percentage = (100-mask_percentage), mask_bs=mask_bs)

        batch_masks = mask_gen.generate().to(device)
        with torch.no_grad():
            x_mask = x[0,0,:,:].unsqueeze(dim=0).expand(mask_bs,-1,-1) - diff_mat.unsqueeze(dim=0).expand(mask_bs,-1,-1) * batch_masks

            x_mask = x_mask.unsqueeze(1)

            h = model.forward_encoder_no_mask(x_mask.to(device))[:,0,:]   # [mask_bs, 768]
            s = pdist(h_star, h).to('cpu')

            # Update R with the mean over the batch
            contribution = abs(batch_masks-1).to('cpu')
            contribution_sum += torch.sum(contribution, dim = (0,1,2))
            tmp_sum = sum([(s[i] * contribution[i]) for i in range(mask_bs)]) / mask_bs

            R = R.add(tmp_sum)
    keep_factor = contribution_sum / (1024*128*mask_bs*num_batches)
    R /= (num_batches * keep_factor)

    #Uncertainty:
    contribution_sum = 0
    for batch_idx in tqdm.trange(num_batches, desc=f"Processing batches (uncertainty)"):
        mask_gen = MaskGenerator(input_shape, mask_type='RELAX', mask_percentage=(100-mask_percentage), mask_bs=mask_bs)

        batch_masks = mask_gen.generate().to(device)
        with torch.no_grad():

            x_mask = x[0,0,:,:].unsqueeze(dim=0).expand(mask_bs,-1,-1) - diff_mat.unsqueeze(dim=0).expand(mask_bs,-1,-1) * batch_masks
            x_mask = x_mask.unsqueeze(1)

            h = model.forward_encoder_no_mask(x_mask.to(device))[:,0,:]   # [mask_bs, 768]
            s = pdist(h_star, h).to('cpu')

            # Update U with the mean over the batch
            vars = [(s[i]-R.to('cpu'))**2 for i in range(len(s))]
            contribution = abs(batch_masks-1).to('cpu')
            contribution_sum += torch.sum(contribution, dim = (0,1,2))
            tmp_sum = sum([(vars[i] * contribution[i]) for i in range(mask_bs)]) / mask_bs

            U = U.add(tmp_sum)

    keep_factor = contribution_sum / (1024*128*mask_bs*num_batches)
    U /= ((num_batches-1)*keep_factor)

    return R, U
