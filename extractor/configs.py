import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', default='./data_text', type=str)
    parser.add_argument('--output_root', default='./data_by_llm', type=str)
    parser.add_argument('--datasets', default=['MAVEN'], type=list, nargs='+')
    parser.add_argument('--model', default='gemini-2.0-flash', type=str)
    parser.add_argument('--candidate', default=1, type=int)
    parser.add_argument('--num_try', default=3, type=int)
    parser.add_argument('--logs_dir', default='./logs/extractor', type=str)
    
    args = parser.parse_args()
    return args