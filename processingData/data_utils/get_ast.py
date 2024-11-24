import os
import re

import javalang
import json

import nltk
import pandas
from tqdm import tqdm
import collections
import sys

from tqdm.contrib import tzip


def tokenize_with_camel_case(token):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]


def tokenize_with_snake_case(token):
    return token.split('_')

def get_FileSize(filePath):
    # filePath = unicode(filePath, 'utf8')
    fsize = os.path.getsize(filePath)
    size = fsize / float(1024)
    return round(size, 2)

def get_name(obj):
    if type(obj).__name__ in ['list', 'tuple']:
        a = []
        for i in obj:
            a.append(get_name(i))
        return a
    elif type(obj).__name__ in ['dict', 'OrderedDict']:
        a = {}
        for k in obj:
            a[k] = get_name(obj[k])
        return a
    elif type(obj).__name__ not in ['int', 'float', 'str', 'bool']:
        return type(obj).__name__
    else:
        return obj


def process_source(path_name):
    num = 0
    no_num = 0
    data = []
    labels = []

    for i, line in enumerate(path_name):

        f_path = line[:-3]
        label = line[-2:-1]
        if os.path.exists(f_path):
            num += 1
            size = get_FileSize(f_path)

            # 如果文件大小为0，则跳过
            if size == 0:
                continue

            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    code = f.read().strip()
            # 使用 'utf-8' 编码打开文件，如果失败，则使用 'gbk' 编码
            except UnicodeDecodeError:
                with open(f_path, 'r', encoding='gbk') as f:
                    code = f.read().strip()

            tokens = list(javalang.tokenizer.tokenize(code))
            tks = []
            for tk in tokens:
                if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                    tks.append('STR_')
                elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                    tks.append('NUM_')
                elif tk.__class__.__name__ == 'Boolean':
                    tks.append('BOOL_')
                else:
                    tks.append(tk.value)
            data.append(" ".join(tks))
            labels.append(label)
        else:
            no_num += 1

    return data, labels, no_num

def get_ast(datas, labels_old):
    global tree
    labels_new = []
    datas_new = []
    ast_data = []
    k = 0
    ign_cnt = 0
    for line, label in tzip(datas, labels_old):
        code = line.strip()

        while code.find(";") != -1:
            # 使用 split 方法拆分字符串
            parts = code.split(";", 1)  # 1 表示只拆分一次

            # 如果成功拆分，提取第二部分（即 "public" 后面的内容）
            if len(parts) > 1:
                code = parts[1].strip()


            tokens = javalang.tokenizer.tokenize(code)
            token_list = list(javalang.tokenizer.tokenize(code))
            length = len(token_list)
            parser = javalang.parser.Parser(tokens)
            try:
                tree = parser.parse_member_declaration()
                break
                # print(tree)
            except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                if code.find(";") == -1:
                    k += 1
                    print(code)
                continue

        flatten = []
        for path, node in tree:
            flatten.append({'path': path, 'node': node})

        ign = False
        outputs = []
        stop = False
        for i, Node in enumerate(flatten):
            d = collections.OrderedDict()
            path = Node['path']
            node = Node['node']
            children = []
            for child in node.children:
                child_path = None
                if isinstance(child, javalang.ast.Node):
                    child_path = path + tuple((node,))
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                            children.append(j)
                if isinstance(child, list) and child:
                    child_path = path + (node, child)
                    for j in range(i + 1, len(flatten)):
                        if child_path == flatten[j]['path']:
                            children.append(j)
            d["id"] = i
            d["type"] = get_name(node)
            if children:
                d["children"] = children
            value = None
            if hasattr(node, 'name'):
                value = node.name
            elif hasattr(node, 'value'):
                value = node.value
            elif hasattr(node, 'position') and node.position:
                for i, token in enumerate(token_list):
                    if node.position == token.position:
                        pos = i + 1
                        value = str(token.value)
                        while pos < length and token_list[pos].value == '.':
                            value = value + '.' + token_list[pos + 1].value
                            pos += 2
                        break
            elif type(node) is javalang.tree.This \
                    or type(node) is javalang.tree.ExplicitConstructorInvocation:
                value = 'this'
            elif type(node) is javalang.tree.BreakStatement:
                value = 'break'
            elif type(node) is javalang.tree.ContinueStatement:
                value = 'continue'
            elif type(node) is javalang.tree.TypeArgument:
                value = str(node.pattern_type)
            elif type(node) is javalang.tree.SuperMethodInvocation \
                    or type(node) is javalang.tree.SuperMemberReference:
                value = 'super.' + str(node.member)
            elif type(node) is javalang.tree.Statement \
                    or type(node) is javalang.tree.BlockStatement \
                    or type(node) is javalang.tree.ForControl \
                    or type(node) is javalang.tree.ArrayInitializer \
                    or type(node) is javalang.tree.SwitchStatementCase:
                value = 'None'
            elif type(node) is javalang.tree.VoidClassReference:
                value = 'void.class'
            elif type(node) is javalang.tree.SuperConstructorInvocation:
                value = 'super'

            if value is not None and type(value) is type('str'):
                d['value'] = value
            if not children and not value:
                # print('Leaf has no value!')
                print(type(node))
                print(code)
                ign = True
                ign_cnt += 1
                # break
            outputs.append(d)
        if not ign:
            ast_data.append(str(json.dumps(outputs)))
            datas_new.append(line)
            labels_new.append(label)

    return ast_data, datas_new, labels_new
