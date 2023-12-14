import torch
import numpy as np
import torch.nn.functional as F

class MaskGenerator:
    def __init__(self, shape, mask_type="original", mask_percentage=30, mask_bs=1, step_length = 1, max_step_size = 1,
                 channels = 3, freq_patch = 4, time_patch = 32):
        self.shape = shape #(time, frequencies) (1024, 128)
        self.mask_type = mask_type
        self.mask_percentage = mask_percentage
        self.mask_bs = mask_bs
        self.step_size = step_length
        self.max_step_size = max_step_size
        self.channels = channels
        self.freq_patch = freq_patch
        self.time_patch = time_patch
        
    def generate(self):
        masks = torch.ones((self.mask_bs, *self.shape))
        patch_size = 8
        patch_size_time = 64
        
        for idx in range(self.mask_bs):
            mask = masks[idx]
            if self.mask_type == "original":
                continue

            elif self.mask_type == "unstructured":
                total_patches = (self.shape[1] // patch_size) * (self.shape[0] // patch_size)
                patches_to_mask = int((self.mask_percentage / 100) * total_patches)

                mask_indices = np.random.choice(total_patches, patches_to_mask, replace=False)

                for index in mask_indices:
                    row = (index // (self.shape[0] // patch_size)) * patch_size
                    col = (index % (self.shape[0] // patch_size)) * patch_size
                    mask[row:row+patch_size, col:col+patch_size] = 0

            elif self.mask_type == "time":
                total_time_patches = self.shape[1] // patch_size
                patches_to_mask = int((self.mask_percentage / 100) * total_time_patches)

                mask_cols = np.random.choice(total_time_patches, patches_to_mask, replace=False)
                for col in mask_cols:
                    mask[:, col*patch_size:(col+1)*patch_size] = 0

            elif self.mask_type == "frequency":
                total_freq_patches = self.shape[0] // patch_size
                patches_to_mask = int((self.mask_percentage / 100) * total_freq_patches)

                mask_rows = np.random.choice(total_freq_patches, patches_to_mask, replace=False)
                for row in mask_rows:
                    mask[row*patch_size:(row+1)*patch_size, :] = 0

            elif self.mask_type == "time+frequency":
                # Time Masking
                tmp_mask = torch.ones((self.shape[0]+self.time_patch, self.shape[1] + self.freq_patch))

                #total_time_patches = self.shape[0] // self.time_patch
                #time_patches_to_mask = int(0.5 * (self.mask_percentage / 100) * total_time_patches)
                #mask_starts = np.random.choice(self.shape[0], time_patches_to_mask, replace=False)
                #for start in mask_starts:
                #    tmp_mask[start:start+self.time_patch, :] = 0
                    #mask[:, col*patch_size_time:(col+1)*patch_size_time] = 0
                mask_sum = 0
                while mask_sum <= (0.5 * (self.mask_percentage/100) * self.shape[0]*self.shape[1]):
                    start = np.random.randint(0, self.shape[0])
                    tmp_mask[start:start+self.time_patch, :] = 0
                    mask_sum = torch.sum(tmp_mask == 0)

                # Frequency Masking
                #total_freq_patches = self.shape[1] // self.freq_patch
                #freq_patches_to_mask = int(0.5 * (self.mask_percentage / 100) * total_freq_patches)
                #mask_starts = np.random.choice(self.shape[1], freq_patches_to_mask, replace=False)
                #for start in mask_starts:
                #    tmp_mask[:, start:start+self.freq_patch] = 0
                    #mask[row*patch_size:(row+1)*patch_size, :] = 0
                while mask_sum <= (self.mask_percentage/100 * self.shape[0]*self.shape[1]):
                    start = np.random.randint(0, self.shape[1])
                    tmp_mask[:, start:start+self.freq_patch] = 0
                    mask_sum = torch.sum(tmp_mask == 0)
                shift_time = np.random.randint(0, self.time_patch)
                shift_freq = np.random.randint(0, self.freq_patch)
                mask[:,:] = tmp_mask[shift_time:shift_time+self.shape[0], shift_freq:shift_freq+self.shape[1]]

            elif self.mask_type == "RELAX":
                ps = 32
                pad_size = (ps,ps,ps,ps)
                #Grid der interpoleres op til size self.shape dimensioner styrer 'formen' af masken
                grid = (torch.rand(self.mask_bs, 1, 10, 10, device='cpu') > self.mask_percentage/100).float()
                grid_up = F.interpolate(grid, size=(1024,1024), mode='bilinear', align_corners=False)
                grid_up = F.pad(grid_up, pad_size, mode='reflect')
                shift_x = np.random.randint(0,ps)
                shift_y = np.random.randint(0,1024-128)
                mask[:,:] = grid_up[0, :, shift_x:shift_x + self.shape[0], shift_y:shift_y + self.shape[1]]

            elif self.mask_type == "Random_walk":
                t = self.shape[0]
                f = self.shape[1]
                ss = self.step_size
                m = self.max_step_size
                width = int(f/100 * (100-self.mask_percentage))
                start = np.random.randint(0, f+1)
                tmp_mask = np.zeros((f+2*width, t+ss))
                for i in range(t//ss + 1):
                    tmp_mask[start+width-(width//2):start+width+(width//2), i*ss:(i+1)*ss] = 1
                    tmp_mask[i*ss:(i+1)*ss, ] = 1
                    if start <= 0:
                        start += np.random.randint(0, m+1)
                    elif start >= f:
                        start -= np.random.randint(0, m+1)
                    else:
                        start += np.random.randint(-m, m+1)
                mask[:,:] = torch.from_numpy(tmp_mask[:t, width:width+f]).float()

            elif self.mask_type == "Random_walk_channels":
                t = self.shape[0]
                f = self.shape[1]
                ss = self.step_size
                m = self.max_step_size
                width = int(f/100 * (100-self.mask_percentage))
                width_c = width // 3 #Enforce equally sized channels
                start = np.random.randint(0, f+1, self.channels)
                tmp_mask = np.zeros((f+2*width, t+ss))
                for c in range(self.channels):
                    for i in range(t//ss + 1):
                        tmp_mask[start[c]+width-(width_c//2):start[c]+width+(width_c//2), i*ss:(i+1)*ss] = 1
                        if start[c] <= 0:
                            start[c] += np.random.randint(0, m+1)
                        elif start[c] >= f:
                            start[c] -= np.random.randint(0, m+1)
                        else:
                            start[c] += np.random.randint(-m, m+1)
                mask[:,:] = torch.from_numpy(tmp_mask[width:width+f, :t]).float()

            else:
                raise ValueError(f"Invalid mask type {self.mask_type}")

        return masks
    

