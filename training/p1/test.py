import os
import torchvision
import torch
from modelA import Generator
from parameters import fix_random_seed
import parameters

def main():
    args = parameters.arg_parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device : {}".format(device))
    fix_random_seed(args.random_seed)

    model_G = Generator().to(device)
    state = torch.load(args.test)
    model_G.load_state_dict(state['state_dict'])
    model_G.eval()
    print(model_G)
    # define some parameters

    with torch.no_grad():
        # Generate 1000 face images (by your script) and evaluate them with the following two metrics: HW2 intro p5
        noise = torch.randn(1000, 100, 1, 1, device=device)
        result = model_G(noise).detach().cpu()
        # filename = os.path.join("./", 'report.png')
        # torchvision.utils.save_image(result[:32], filename, nrow=8)
        for i in range(result.shape[0]):
            img = result[i]
            filename = os.path.join(args.save_test_result_dir, '{:04d}.png'.format(i + 1))
            torchvision.utils.save_image(img, filename, normalize=True)


if __name__ == '__main__':
    main()
