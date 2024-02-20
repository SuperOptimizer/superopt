
import pandas as pd
import os
import bs4
import html_to_json
import re


ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..'))

def decode_modrm(modrm):
    assert 0 <= modrm < 256
    mod = modrm >> 6
    reg = (modrm >> 3) & 0x7
    rm = modrm & 0x7


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_word_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

optree = Trie()

def parse():
    txt  = open(os.path.join(ROOTDIR,'doc','shellstorm','shell-storm.org','x86doc','index.html'),'r').read()
    txt = txt.replace("<sub>","").replace("</sub>","").replace("<sup>","").replace("</sup>","")
    asdf = html_to_json.convert(txt,capture_element_values=True, capture_element_attributes=True)
    for entry in asdf['html'][0]['body'][0]['div'][0]['p'][1]['table'][0]['tr']:
        if 'td' in entry:
            instr = entry['td'][0]['a'][0]['_value']
            operands = entry['td'][0].get('_value', '')
            if operands == '':
                num_operands = 0
            else:
                num_operands = operands.count(',') + 1
                if num_operands == 0:
                    dst = src1 = ''
                if num_operands == 1:
                    dst = operands.split(',')[0]
                    src1 = ''
                elif num_operands == 2:
                    dst,src1 = operands.split(',')
                    print()
                elif num_operands == 3:
                    dst,src1,src2 = operands.split(',')
                elif num_operands == 4:
                    dst,src1,src2,src3 = operands.split(',')
                else:
                    assert False

            href = entry['td'][0]['a'][0]['_attributes']['href'].replace('./','')
            pg  = open(os.path.join(ROOTDIR,'doc','shellstorm','shell-storm.org','x86doc',href),'r').read()
            pg = pg.replace("<sub>","").replace("</sub>","").replace("<sup>","").replace("</sup>","").replace('<em>','').replace('</em>','')
            #shellstorm has some incorrect formatting, like '(cid:197)' instead of ':='. it also uses ← so lets replace those
            pg = pg.replace('←',':=').replace('(cid:197)',':=')
            qwer = html_to_json.convert(pg,capture_element_values=True, capture_element_attributes=True)
            if 'pre' in qwer['html'][0]['body'][0]:
                if len(qwer['html'][0]['body'][0]['pre']) == 1:
                    operation = qwer['html'][0]['body'][0]['pre'][0].get('_value','')
                else:
                    operation = [e.get('_value','') for e in qwer['html'][0]['body'][0]['pre']]
            else:
                operation = ''
            print()


            op = entry['td'][1]['_value']

            ext = entry['td'][2].get('_value', '')
            desc = entry['td'][3]['_value']
            for i,x in enumerate(op.replace(' +',' ').replace('+ ','').split()):
                try:
                    code = int(x,16)
                except:
                    pass

            print(instr,operands,op,ext,desc)

    #print(asdf)




if __name__ == '__main__':
    parse()