"""
This is code is copied from 
http://rosettacode.org/wiki/Huffman_coding#Python
"""
from heapq import heappush, heappop, heapify

def Huffman_code(symb2freq):
    if len(symb2freq) == 1:
        code = list(symb2freq)
        code.append('0')
        return [code]
    else:
        """Huffman encode the given dict mapping symbols to weights"""
        heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
        heapify(heap)
        while len(heap) > 1:
            lo = heappop(heap)
            hi = heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def encode_Huffman(symb2freq, lmbda):
    code = Huffman_code(symb2freq)
    return [[c[0], lmbda*len(c[1])] for c in code]
