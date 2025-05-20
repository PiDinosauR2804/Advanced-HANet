import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_root', default='./data/data_text', type=str)

args, _ = parser.parse_known_args()
print(args.input_root)
print(args['input_root'])
