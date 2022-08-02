import cv2
import numpy as np
import os


HuffmanCode = {}

class Node:
    def __init__(self, value, symbol, left=None, right=None):
        self.code = ''
        self.value = value
        self.symbol = symbol
        self.left = left
        self.right = right

def pixel_dic_list(list):
    pixel_dic = {}      #創造一個dictionary，統計數字出現的頻率
    for i in list:
        if i not in pixel_dic.keys():
            pixel_dic[i] = 1
        else:
            pixel_dic[i] = pixel_dic[i] + 1
    return pixel_dic

def HuffmanTree_build(node_list):
    while len(node_list) > 1:      #開始合併
        node_list = sorted(node_list, key=lambda x : x.value)
        #for i in node_list:
           # print(i.symbol, i.value)
        #print()

        rightNode = node_list[0]
        leftNode = node_list[1]

        new_Node = Node(value=leftNode.value+rightNode.value, symbol=leftNode.symbol+rightNode.symbol, left = leftNode, right = rightNode)

        node_list.remove(leftNode)
        node_list.remove(rightNode)
        node_list.append(new_Node)

    return node_list


def Code_build(node):
    if(node.left):
        node.left.code = node.code + '1'
        Code_build(node.left)
    if(node.right):
        node.right.code = node.code + '0'
        Code_build(node.right)
    
    if(not node.left and not node.right):   ##代表葉節點，就是沒有合併的那些點
        HuffmanCode[node.symbol] = node.code

def HuffmanCode_sort():
    global HuffmanCode
    HuffmanCode = sorted(HuffmanCode.items())
    HuffmanCode = dict(HuffmanCode)

def main():
    file_name = os.path.join(os.path.dirname(__file__), 'lena.bmp')
    assert os.path.exists(file_name)
    img = cv2.imread(file_name)
    h, w = img.shape[:2]
    img2 = np.zeros((h, w), dtype=np.uint16)
    pixel_list = []
    for i in range(h):
        for j in range(w):
            pixel_list.append(img[i, j][0])
            #print(img[i, j][0])
    pixel_dic = pixel_dic_list(pixel_list)
    #print(pixel_dic)
    
    node_list = []
    for symbol in pixel_dic.keys():
        node_list.append(Node(value=pixel_dic.get(symbol), symbol = symbol))

    tree_head = HuffmanTree_build(node_list)[0]     ##求出頂部的點

    #print(tree_head.right.symbol, tree_head.right.value)

    Code_build(tree_head)

    HuffmanCode_sort()

    print(HuffmanCode)

    

main()
