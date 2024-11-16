import fid
import torch
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fid_score(out_folder, gt_folder):
    score = fid.calculate_fid_given_paths([out_folder, gt_folder],
                                          batch_size = 50,
                                          device = device,
                                          dims = 2048)
    return score

print(fid_score("./test", "../hw2_data/face/train/"))