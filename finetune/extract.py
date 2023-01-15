# -*- coding: UTF-8 -*-

''' Extract data from file by the special keyword '''

import argparse
from os import path
from re import search


class DataExtractor(object):
    ''' DataExtrator class '''

    def __init__(self, infile, keyword, outfile):
        '''
        构造函数

        infile：输入文件名
        keyword：目标数据前面的关键字
        outfile：输出文件名
        '''

        self.infile = infile
        self.keyword = keyword
        self.outfile = outfile

    def data_after_keyword(self):
        ''' Extract data from infile after the keyword. '''

        try:
            data = []
            patt = '%s (\d+\.?\d?)' % self.keyword  # 使用正则表达式搜索数据
            with open(self.infile, 'r') as fi:
                with open(self.outfile, 'w') as fo:
                    for eachLine in fi:
                        s = search(patt, eachLine)
                        if s is not None:
                            fo.write(s.group(1) + '\n')
                            data.append(float(s.group(1)))
            return data
        except IOError:
            print(
                "Open file [%s] or [%s] failed!" % (self.infile, self.outfile))
            return False


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser(description='Extract data from file.')
    parser.add_argument('infile', default='shishilog.log', help='input file which contains data')
    parser.add_argument('keyword', default='test_rmse:', help='keywords for find data')
    # parser.add_argument('-o', '--outfile',
    #                     default=path.basename(__file__).split('.')[0] + '.out',
    #                     help='output file for save data')
    parser.add_argument('--outfile',
                        default='shishilog.out',
                        help='output file for save data')

    args = parser.parse_args()  # 命令行参数解析
    infile = args.infile
    outfile = args.outfile
    keyword = args.keyword

    extractor = DataExtractor(infile, keyword, outfile)
    ret = extractor.data_after_keyword()
    if ret:
        print("Export data from file[%s] after keyword[%s] successed!" % (
            infile, keyword))
    else:
        print("Export data from file[%s] after keyword[%s] failed!" % (
            infile, keyword))


if __name__ == '__main__':
    main()

