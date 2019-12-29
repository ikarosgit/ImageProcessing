import os
import argparse

from solver import Solver

def main(args):

    solver = Solver(
        args.imori_path,
        args.imori_noise_path)

    solver.solve(args.question)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("question", type=int)
    parser.add_argument("--imori_path", type=str, default="data/imori.jpg")
    parser.add_argument("--imori_noise_path", type=str, default="data/imori_noise.jpg")
    args = parser.parse_args()

    main(args)
