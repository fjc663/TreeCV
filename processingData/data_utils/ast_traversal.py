import json

import pandas


def SBT_(cur_root_id, node_list, key='type'):
    cur_root = node_list[cur_root_id]
    tmp_list = ["("]

    str = cur_root[key]
    tmp_list.append(str)

    if 'children' in cur_root:
        chs = cur_root['children']
        for ch in chs:
            tmp_list.extend(SBT_(ch, node_list))
    tmp_list.append(")")
    tmp_list.append(str)
    return tmp_list


def get_sbt_structure(asts, key):
    sbt_data = []
    for a in asts:
        a = json.loads(a)
        ast_sbt = SBT_(0, a, key)
        sbt_data.append(' '.join(ast_sbt))


    return sbt_data
