import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from parse_args import parse_args
from utils_remove_target_source_history_pre import *
from ddpm_conditional_remove_target_pre import Diffusion


def main():
    # Parse args, seed RNGs, and run pretraining loop.
    args = parse_args()
    set_seed(args.seed, reproducible=True)

    diffuser = Diffusion(
        noise_steps=args.noise_steps,
        lamda=args.lamda,
    )
    diffuser.prepare(args)
    diffuser.fit(args)
    

if __name__ == '__main__':
    main()