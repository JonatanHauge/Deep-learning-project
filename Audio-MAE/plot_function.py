import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import torch.nn as nn


def ALL_plot_AUDIO_MAE(importance_mat, uncertainty_mat, model, x):
    """importance_mat shape: (timesteps, frequency), (1024, 128)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pdist = nn.CosineSimilarity(dim=1)
    t = importance_mat.shape[0]
    f = importance_mat.shape[1]
    min_x = x.min().to(device)
    insert_list = []
    delete_list = []
    percent_idx_list = 100*np.cumsum([t for _ in range(f)]) / (t*f)

    with torch.no_grad():
        h_star = model.forward_encoder_no_mask(x.to(device))[:,1:,:].mean(dim=1) # [1, 768]
        imp_mat_flat = importance_mat.flatten()
        
        for i in tqdm.trange(f, desc=f"Processing pixels for AUDIO-MAE"):
            top_k = torch.topk(imp_mat_flat, t*(i+1))
            replace_mask_top = (importance_mat > top_k.values[t*(i+1)-1])
            replace_mask_bottom = (importance_mat <= top_k.values[t*(i+1)-1])
            
            #Insertion:
            new_x_ins = replace_mask_top.to(device) * x[0,0,:,:] + min_x * replace_mask_bottom.to(device)
            new_x_ins = new_x_ins.unsqueeze(0)
            new_x_ins = new_x_ins.unsqueeze(0)
            h_ins = model.forward_encoder_no_mask(new_x_ins.to(device))[:,1:,:].mean(dim=1)
            s_ins = pdist(h_star, h_ins).to('cpu')
            insert_list.append(s_ins)

            #Deletion:
            new_x_del = replace_mask_bottom.to(device) * x[0,0,:,:] + min_x * replace_mask_top.to(device)
            new_x_del = new_x_del.unsqueeze(0)
            new_x_del = new_x_del.unsqueeze(0)
            h_del = model.forward_encoder_no_mask(new_x_del.to(device))[:,1:,:].mean(dim=1)
            s_del = pdist(h_star, h_del).to('cpu')
            delete_list.append(s_del)

    fig = plt.figure(figsize=(10, 3))

    #Define axis positions
    ax1 = fig.add_axes([0.05, 0.5, 0.6, 1])
    ax2 = fig.add_axes([0.05, 0.1, 0.6, 1])
    ax3 = fig.add_axes([0.7, 0.53, 0.3, 0.6])
    ax = [ax1, ax2, ax3]
    
    #Plotting importance and uncertainty
    ax1.imshow(x[0,0,:,:].T.to('cpu'), origin = 'lower',cmap = 'gray')
    ax2.imshow(x[0,0,:,:].T.to('cpu'), origin = 'lower',cmap = 'gray')
    im0 = ax1.imshow(importance_mat.T, origin = 'lower', cmap='bwr', alpha=0.7)
    im1 = ax2.imshow(uncertainty_mat.T, origin = 'lower', cmap='bwr', alpha=0.7)
    fig.subplots_adjust(right=0.8)

    #Positioning colorbar
    cbar_ax = fig.add_axes([0.66, 0.5, 0.015, 0.6])
    cbar = fig.colorbar(im0, cax=cbar_ax)
    cbar.ax.set_yticks([])
    ax1.set_title('Importance')
    ax2.set_title('Uncertainty')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    #Model name on the left of the plot
    fig.text(0.03, 0.8, 'AUDIO-MAE', va='center', rotation='vertical', fontsize=12)
            
    #Plotting insertion/deletion
    ax3.plot(percent_idx_list, insert_list, label='Insertion')
    ax3.grid(which='both', linestyle='--', linewidth=0.5)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('Cosine similarity')
    ax3.set_xlabel('percentage of pixels inserted/deleted')
    ax3.plot(percent_idx_list, delete_list, label='Deletion', color='red')
    ax3.legend()
    plt.show()
    
    return fig, ax