# coding:utf-8
import argparse
import os

import pandas

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tqdm import tqdm

import re
import tokenize
from io import BytesIO

from tqdm.contrib import tzip

from data_utils.get_ast import process_source, get_ast
from data_utils.ast_traversal import get_sbt_structure
from data_utils.data_loader import load_data
from data_utils.word2vec import train_vord2vec_and_save, word2vec_net
from data_utils.seq2img import TokenVisDataset


def replace_newline_and_indent(code):
    # 将缩进替换为"INDENT "
    i = 10
    for _ in range(10):
        str = ''
        for _ in range(i):
            str += 'INDENT '
        pattern = '^(\s{%d})' % (i * 4)
        code = re.sub(pattern, str, code.strip(), flags=re.MULTILINE)
        i -= 1
    # 将换行替换为"NEWLINE "
    # code = code.replace('\n', 'NEWLINE ')
    return code


def tokenize_code(code_str, newline=False, comment=False):
    if comment:
        # 使用正则表达式匹配并删除C语言注释
        # 匹配 /* ... */ 形式的多行注释
        code = re.sub(r'/\*.*?\*/', '', code_str, flags=re.DOTALL)

        # 匹配 // 形式的单行注释
        code_str = re.sub(r'//.*', '', code)

    code_str = code_str.strip()
    # 将代码字符串转换为字节流
    code_bytes = code_str.encode('utf-8')

    # 准备一个列表来存储分词后的结果
    tokenized_code = []

    # 使用tokenize模块进行分词
    for token in tokenize.tokenize(BytesIO(code_bytes).readline):
        _, token_value, _, _, _ = token

        if newline:
            if token_value == '\n':
                # 将换行替换为"NEWLINE"
                tokenized_code.append('NEWLINE')
            else:
                tokenized_code.append(token_value)
        else:
            if token_value != '\n':
                tokenized_code.append(token_value)

    return tokenized_code[1:]


def get_FileSize(filePath):
    # filePath = unicode(filePath, 'utf8')
    fsize = os.path.getsize(filePath)
    size = fsize / float(1024)
    return round(size, 2)


def read_file(file_name):
    word2vec_data_id = []
    word2vec_data = []
    id = 0
    with open(file_name, 'r') as f:
        for line in tqdm(f.readlines()):
            word2vec_data_id.append(id)
            word2vec_data.append(line.split())
            id += 1

    return word2vec_data_id, word2vec_data


def train_word2vec(original_dataset, model_filename='word2vec_code.pth', blk_width=4):
    # train and save word2vec model
    print("Loading pretrain data...")

    codes = []
    x = 0
    for code in original_dataset:
        try:
            codes.append(tokenize_code(replace_newline_and_indent(code), newline=True, comment=True))
        except:
            x += 1

    # debug
    # lst = []
    # count = 0
    # for i in codes:
    #     lst.append(len(i))
    #     if len(i) < 625:
    #         count += 1
    #
    # print('max', max(lst))
    # print('min', min(lst))
    # print('avg', sum(lst) / len(codes))
    # print('count', count)
    # print('sum', len(codes))
    # print('x', x)
    #
    # exit(0)

    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data(codes, batch_size, max_window_size,
                                 num_noise_words, num_workers=5)
    lr, num_epochs, embed_size = 0.002, 2, blk_width * blk_width

    net = word2vec_net(len(vocab), embed_size)

    if not os.path.isfile(model_filename):
        print('Pretrain model not found, start training...')
        train_vord2vec_and_save(net, data_iter, lr, num_epochs, save_path=model_filename)
    else:
        print('Pretrain model founded, loading state dict...')
        net.load_state_dict(torch.load(model_filename))

    return net, vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--blk_width', type=int, default=4, help='块宽度，默认值为4')
    parser.add_argument('--img_type', type=str, default='tsbt', help='图像类型，默认值为tsbt')

    args = parser.parse_args()

    blk_width = args.blk_width
    img_type = args.img_type
    
    total_code_data = {}
    total_tsbt_data = {}
    total_vsbt_data = {}
    total_filename = {}
    total_labels = {}
    train_data_code = []
    train_data_tsbt = []
    train_data_vsbt = []
    data_path = '../data/txt'
    save_path = '../data/img_' + img_type + '_' + str(blk_width) + 'w'
    no_nums = 0
    # 遍历目录下的每个文件
    for txtfile in os.listdir(data_path):
        # 跳过系统文件
        if txtfile in ['.DS_Store', '._ant-1.7.txt', '._.DS_Store', '._jEdit-4.0.txt']:
            continue

        txt_path = os.path.join(data_path, txtfile)
        txt_key = txtfile.split('.txt')[0]
        with open(txt_path, 'r') as f:
            path_name = f.readlines()
            # pre-process the source code: strings -> STR_, numbers-> NUM_, Booleans-> BOOL_
            code_data, labels, no_num = process_source(path_name)
            no_nums += no_num

            if img_type == 'token' or img_type == 'grid' or img_type == 'token_tvsbt' or img_type == 'WysiWiM' or img_type == 'ascii':
                total_code_data[txt_key] = code_data
                train_data_code.extend(code_data)
            if img_type == 'tsbt' or img_type == 'vsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
                # generate ast file for source code
                ast_data, code_data, labels = get_ast(code_data, labels)
                if img_type == 'tsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
                    tsbt_data = get_sbt_structure(ast_data, 'type')
                    total_tsbt_data[txt_key] = tsbt_data
                    train_data_tsbt.extend(tsbt_data)
                if img_type == 'vsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
                    vsbt_data = get_sbt_structure(ast_data, 'value')
                    total_vsbt_data[txt_key] = vsbt_data
                    train_data_vsbt.extend(vsbt_data)
            if img_type == 'DTLDP':
                filenames = []
                for line in path_name:
                    f_path = line[:-3]
                    if os.path.exists(f_path):
                        size = get_FileSize(f_path)
                        # 如果文件大小为0，则跳过
                        if size == 0:
                            continue

                        filenames.append(f_path)

                total_filename[txt_key] = filenames

            total_labels[txt_key] = labels

    print('no_nums: ', no_nums)

    if img_type == 'token_tvsbt':
        net_code, vocab_code = train_word2vec(train_data_code,
                                              model_filename='word2vec_code.pth', blk_width=blk_width)
        net_tsbt, vocab_tsbt = train_word2vec(train_data_tsbt,
                                              model_filename='word2vec_tsbt.pth', blk_width=blk_width)
        net_vsbt, vocab_vsbt = train_word2vec(train_data_vsbt,
                                              model_filename='word2vec_vsbt.pth', blk_width=blk_width)
        dataset = TokenVisDataset(net_code[0], net_tsbt[0], net_vsbt[0], blk_width, img_type=img_type)
        for key, code_data, tsbt_data, vsbt_data, labels in tzip(total_code_data.keys(), total_code_data.values(),
                                                                 total_tsbt_data.values(), total_vsbt_data.values(),
                                                                 total_labels.values()):
            dataset.build(key, labels, save_path, code_data, tsbt_data, vsbt_data, vocab_code, vocab_tsbt, vocab_vsbt)

    elif img_type == 'token' or img_type == 'grid':
        net_code, vocab_code = train_word2vec(train_data_code,
                                              model_filename='word2vec_code.pth', blk_width=blk_width)
        dataset = TokenVisDataset(embed_code=net_code[0], blk_width=blk_width, img_type=img_type)
        for key, code_data, labels in tzip(total_code_data.keys(), total_code_data.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_code=code_data, vocab_code=vocab_code)

    elif img_type == 'tsbt':
        net_tsbt, vocab_tsbt = train_word2vec(train_data_tsbt,
                                              model_filename='word2vec_tsbt.pth', blk_width=blk_width)
        dataset = TokenVisDataset(embed_tsbt=net_tsbt[0], blk_width=blk_width, img_type=img_type)
        for key, tsbt_data, labels in tzip(total_tsbt_data.keys(), total_tsbt_data.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_tsbt=tsbt_data, vocab_tsbt=vocab_tsbt)

    elif img_type == 'vsbt':
        net_vsbt, vocab_vsbt = train_word2vec(train_data_vsbt,
                                              model_filename='word2vec_vsbt.pth', blk_width=blk_width)
        dataset = TokenVisDataset(embed_vsbt=net_vsbt[0], blk_width=blk_width, img_type=img_type)
        for key, vsbt_data, labels in tzip(total_vsbt_data.keys(), total_vsbt_data.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_vsbt=vsbt_data, vocab_vsbt=vocab_vsbt)

    elif img_type == 'tvsbt':
        net_tsbt, vocab_tsbt = train_word2vec(train_data_tsbt,
                                              model_filename='word2vec_tsbt.pth', blk_width=blk_width)
        net_vsbt, vocab_vsbt = train_word2vec(train_data_vsbt,
                                              model_filename='word2vec_vsbt.pth', blk_width=blk_width)
        dataset = TokenVisDataset(embed_tsbt=net_tsbt[0], embed_vsbt=net_vsbt[0], blk_width=blk_width, img_type=img_type)
        for key, tsbt_data, vsbt_data, labels in tzip(total_tsbt_data.keys(),total_tsbt_data.values(), total_vsbt_data.values(),
                                                                 total_labels.values()):
            dataset.build(key, labels, save_path, inputs_tsbt=tsbt_data, inputs_vsbt=vsbt_data, vocab_tsbt=vocab_tsbt, vocab_vsbt=vocab_vsbt)

    elif img_type == 'DTLDP':
        dataset = TokenVisDataset(img_type=img_type)
        for key, filename, labels in tzip(total_filename.keys(), total_filename.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_filename=filename)

    elif img_type == 'WysiWiM':
        dataset = TokenVisDataset(img_type=img_type)
        for key, code_data, labels in tzip(total_code_data.keys(), total_code_data.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_code=code_data)

    elif img_type == 'ascii':
        dataset = TokenVisDataset(img_type=img_type)
        for key, code_data, labels in tzip(total_code_data.keys(), total_code_data.values(), total_labels.values()):
            dataset.build(key, labels, save_path, inputs_code=code_data)
