
def load_vectors(file_path):
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
    return data
