import binascii
import csv
import math
import os.path
import re

import cv2
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from torchvision import transforms
from tqdm.contrib import tzip

import numpy
import numpy as np


def tokenize_code(code):
    # Matching Identifiers, Operators, Constants, Line Breaks, Tab Symbols,
    # and Spaces in Java Code with Regular Expressions
    pattern = r'\b\w+\b|[-+*/=<>()[\]{};]|[\n\t]'
    tokens = re.findall(pattern, code)
    return tokens

class TokenVisDataset:

    def __init__(self, embed_code=None, embed_tsbt=None, embed_vsbt=None, blk_width=4, img_type='token_tvsbt'):
        # embed embedding层
        self.blk_width = blk_width
        self.img_type = img_type

        if img_type == 'token' or img_type == 'grid' or img_type == 'token_tvsbt':
            self.embed_weight_code = embed_code.weight.data
        if img_type == 'tsbt' or img_type == 'vsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
            if img_type == 'tsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
                self.embed_weight_tsbt = embed_tsbt.weight.data
            if img_type == 'vsbt' or img_type == 'token_tvsbt' or img_type == 'tvsbt':
                self.embed_weight_vsbt = embed_vsbt.weight.data

    def token2block(self, embed_weight_code, token, vocab):
        # 找嵌入向量，reshape
        # 输入：token
        # 输出：像素块
        vec = embed_weight_code[vocab[token]]
        return vec.reshape(self.blk_width, self.blk_width).cpu().numpy()

    @staticmethod
    def get_img(codeimg, img_size=(224, 224)):
        data_transforms = transforms.Compose([
            transforms.Resize(330),
            transforms.CenterCrop(img_size)
        ])

        # 将矩阵转换为图像格式（灰度图像）
        # 计算最小值和最大值
        min_val = np.min(codeimg)
        max_val = np.max(codeimg)

        # 缩放数组的值到0到255之间
        codeimg = 255 * (codeimg - min_val) / (max_val - min_val)

        # 将NumPy数组转换为PIL图像
        pil_image = Image.fromarray(codeimg.astype(np.uint8))
        pil_image = data_transforms(pil_image)

        return pil_image

    # 拼接
    @staticmethod
    def joinery(s, split_arr_list):
        concatenated_arr_list = []
        for i in range(s):
            concat_arr = np.concatenate(split_arr_list[i].squeeze(0), axis=1)
            concatenated_arr_list.append(concat_arr)

        return concatenated_arr_list

    @staticmethod
    def get_FileSize(filePath):
        fsize = os.path.getsize(filePath)
        size = fsize / float(1024)
        return round(size, 2)

    def getAscii_img(self, code):
        content = str(code).encode('utf-8')
        hexst = binascii.hexlify(content)
        fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])

        x = len(fh)
        x_width = math.ceil(math.sqrt(x))
        y = x_width * x_width - x

        fh = np.pad(fh, (0, y), 'constant', constant_values=0)
        img = np.array(fh).reshape(x_width, x_width)

        return self.get_img(img)

    def getWysiWiM_img(self, code, fonts='/usr/share/fonts/truetype/freefont/FreeSerif.ttf'):
        code = str(code).replace("\t", "    ")
        background = (255, 255, 255)

        fontsize = 14
        font = ImageFont.truetype(fonts, fontsize)

        # 使用textbbox替代textsize
        bbox = ImageDraw.Draw(Image.new('RGBA', (1, 1), background)).textbbox((0, 0), code, font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        image = Image.new('RGBA', (int(width * 1.1), int(height * 1.1)), background)
        draw = ImageDraw.Draw(image)
        draw.text((10, 2), code, fill='black', font=font)
        image = image.convert('L')
        # image = np.array(image)[np.newaxis, :, :]

        resize = transforms.Resize([224, 224])

        return resize(image)

    def getDTLDP_img(self, filename):
        size = self.get_FileSize(filename)
        n = 1
        if (size == 0):
            return 0
        if size < 10:
            width = int(32 * n)
        elif size < 30:
            width = int(64 * n)
        elif size < 60:
            width = int(128 * n)
        elif size < 100:
            width = int(256 * n)
        elif size < 200:
            width = int(384 * n)
        elif size < 500:
            width = int(512 * n)
        elif size < 1000:
            width = int(768 * n)
        else:
            width = int(1024 * n)

        with open(filename, 'rb') as f:
            content = f.read()
        hexst = binascii.hexlify(content)
        fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])

        end = len(fh) - len(fh) % 3

        # b = fh[0:end:3]
        # g = fh[1:end:3]
        # r = fh[2:end:3]

        r = fh[0:end:3]
        b = fh[1:end:3]
        g = fh[2:end:3]

        # r = fh[0:end:3]
        # g = fh[1:end:3]
        # b = fh[2:end:3]
        #
        # b = fh[0:end:3]
        # r = fh[1:end:3]
        # g = fh[2:end:3]
        #
        # g = fh[0:end:3]
        # b = fh[1:end:3]
        # r = fh[2:end:3]
        #
        # g = fh[0:end:3]
        # r = fh[1:end:3]
        # b = fh[2:end:3]

        img2 = cv2.merge([b, g, r])
        img1 = img2[:len(b) - len(b) % width]
        img = np.reshape(img1, (width, len(b) // width, 3))

        return self.get_img(img)

    def get_gridImg(self, embed_weight_code, vocab_code, codes, img_size):

        space_vec = numpy.zeros(self.blk_width * self.blk_width)
        space_vec = space_vec.reshape(space_vec.shape[0], 1, 1)

        # 一段代码按行转成一个个Token列表
        code_lines = [tokenize_code(line) for line in codes.split('\n')]

        # 构建codegrid
        # codegrid格式：一个像素点一个向量
        codegrid = []
        max_width = 0

        for line in code_lines:
            grid_line = []
            for token in line:
                token_len = len(token)
                if token == "\t":
                    token_len = 2
                    token_vec = self.token2block(embed_weight_code, token, vocab_code).reshape(self.blk_width * self.blk_width, 1, 1)
                    grid_line = grid_line + [token_vec] * token_len
                else:
                    token_vec = self.token2block(embed_weight_code, token, vocab_code).reshape(self.blk_width * self.blk_width, 1, 1)
                    grid_line = grid_line + [token_vec] * token_len
                    grid_line = grid_line + [space_vec]

            if max_width < len(grid_line):
                max_width = len(grid_line)
            # 添加转换好的一行
            codegrid.append(grid_line)

        codegrid = [line + [space_vec] * (max_width - len(line)) for line in codegrid]

        grid_row = [np.concatenate(line, axis=2) for line in codegrid]
        grid_npy = np.concatenate(grid_row, axis=1)

        grid_img = np.zeros((self.blk_width * self.blk_width, img_size[0], img_size[1]))
        for i in range(grid_npy.shape[0]):
            for j in range(grid_npy.shape[1]):
                grid_img[i, :, :] = cv2.resize(grid_npy[i, :, :], (img_size[0], img_size[1]))

        return grid_img

    def get_pilImg(self, embed_weight, vocab, tokens, img_size):
        # line: line for line in code_lines
        block_lines = [self.token2block(embed_weight, code, vocab) for code in tokens]

        empty_block = numpy.zeros([self.blk_width, self.blk_width])

        s = math.ceil(math.sqrt(len(block_lines)))
        block_lines = block_lines + [empty_block] * (s * s - len(block_lines))

        # s = math.floor(math.sqrt(len(block_lines)))
        # block_lines = block_lines[:s*s]

        # 拼接
        blocks = np.array(block_lines).reshape(s, s, self.blk_width, self.blk_width)

        split_arr_list = np.array_split(blocks, s)

        concatenated_arr_list = self.joinery(s, split_arr_list)

        # 将拼接后的25个3x75的数组在第一维拼接得到75x75的数组
        codeNpy = np.concatenate(concatenated_arr_list, axis=0)

        codeimg = self.get_img(codeNpy, img_size)

        return codeimg

    def visualize(self, codes=None, tsbts=None, vsbts=None, vocab_code=None, vocab_tsbt=None, vocab_vsbt=None, save_path=None, filename=None, img_size=(224, 224)):

        if self.img_type == 'token_tvsbt':
            code_image = self.get_pilImg(self.embed_weight_code, vocab_code, codes, img_size)
            tsbt_image = self.get_pilImg(self.embed_weight_tsbt, vocab_tsbt, tsbts, img_size)
            vsbt_image = self.get_pilImg(self.embed_weight_vsbt, vocab_vsbt, vsbts, img_size)

            # 创建一个新的RGB图像
            rgb_image = Image.new('RGB', img_size)

            # 将灰度图转为三通道图并拼接
            rgb_image.paste(Image.merge('RGB', (code_image, tsbt_image, vsbt_image)), (0, 0))

            rgb_image.save(save_path, format="PNG", compression=None)
            # rgb_image.show()
        elif self.img_type == 'token':
            code_image = self.get_pilImg(self.embed_weight_code, vocab_code, codes, img_size)
            code_image.save(save_path, format="PNG", compression=None)
        elif self.img_type == 'tsbt':
            tsbt_image = self.get_pilImg(self.embed_weight_tsbt, vocab_tsbt, tsbts, img_size)
            tsbt_image.save(save_path, format="PNG", compression=None)
        elif self.img_type == 'vsbt':
            vsbt_image = self.get_pilImg(self.embed_weight_vsbt, vocab_vsbt, vsbts, img_size)
            vsbt_image.save(save_path, format="PNG", compression=None)
        if self.img_type == 'tvsbt':
            tsbt_image = self.get_pilImg(self.embed_weight_tsbt, vocab_tsbt, tsbts, img_size)
            vsbt_image = self.get_pilImg(self.embed_weight_vsbt, vocab_vsbt, vsbts, img_size)

            # 创建一个新的图像，大小与原图相同，但通道数为两通道
            combined_image = Image.new('RGB', img_size)
            # 将tsbt_image作为第一通道
            combined_image.paste(tsbt_image, (0, 0))
            # 将vsbt_image作为第二通道
            combined_image.paste(vsbt_image, (0, 0))

            combined_image.save(save_path, format="PNG", compression=None)
        elif self.img_type == 'grid':
            grid_image = self.get_gridImg(self.embed_weight_code, vocab_code, codes, img_size)
            np.save(save_path, grid_image)
        elif self.img_type == 'DTLDP':
            DTLDP_image = self.getDTLDP_img(filename)
            DTLDP_image.save(save_path, format="PNG", compression=None)
        elif self.img_type == 'WysiWiM':
            WysiWiM_image = self.getWysiWiM_img(codes)
            WysiWiM_image.save(save_path, format="PNG", compression=None)
        elif self.img_type == 'ascii':
            ascii_image = self.getAscii_img(codes)
            ascii_image.save(save_path, format="PNG", compression=None)

    def build(self, txtname, labels, save_img_path, inputs_code=None, inputs_tsbt=None, inputs_vsbt=None, vocab_code=None, vocab_tsbt=None, vocab_vsbt=None, inputs_filename=None):
        # Build entire dataset from inputs
        print("Creating visual representation...")

        save_csv_data = []
        # 使用os.makedirs()创建文件夹
        folder_path = os.path.join(save_img_path, txtname)
        os.makedirs(folder_path, exist_ok=True)

        if self.img_type == 'token_tvsbt':
            for id, code, tsbt, vsbt, label in tzip(range(len(inputs_code)), inputs_code, inputs_tsbt, inputs_vsbt,
                                                    labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")
                save_csv_data.append([save_path, label])
                self.visualize(code, tsbt, vsbt, vocab_code, vocab_tsbt, vocab_vsbt,
                               save_path=save_path)

        elif self.img_type == 'token' or self.img_type == 'grid':
            for id, code, label in tzip(range(len(inputs_code)), inputs_code, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                if self.img_type == 'grid':
                    save_path = os.path.join(folder_path, f"{txtname}_{id}.npy")

                save_csv_data.append([save_path, label])
                self.visualize(codes=code, vocab_code=vocab_code, save_path=save_path)

        elif self.img_type == 'tsbt':
            for id, tsbt, label in tzip(range(len(inputs_tsbt)), inputs_tsbt, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                save_csv_data.append([save_path, label])
                self.visualize(tsbts=tsbt, vocab_tsbt=vocab_tsbt, save_path=save_path)

        elif self.img_type == 'vsbt':
            for id, vsbt, label in tzip(range(len(inputs_vsbt)), inputs_vsbt, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                save_csv_data.append([save_path, label])
                self.visualize(vsbts=vsbt, vocab_vsbt=vocab_vsbt, save_path=save_path)

        elif self.img_type == 'tvsbt':
            for id, tsbt, vsbt, label in tzip(range(len(inputs_tsbt)), inputs_tsbt, inputs_vsbt,
                                                    labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")
                save_csv_data.append([save_path, label])
                self.visualize(tsbts=tsbt, vsbts=vsbt, vocab_tsbt=vocab_tsbt, vocab_vsbt=vocab_vsbt,
                               save_path=save_path)

        elif self.img_type == 'DTLDP':
            for id, filename, label in tzip(range(len(inputs_filename)), inputs_filename, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                save_csv_data.append([save_path, label])
                self.visualize(filename=filename, save_path=save_path)

        elif self.img_type == 'WysiWiM':
            for id, code, label in tzip(range(len(inputs_code)), inputs_code, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                save_csv_data.append([save_path, label])
                self.visualize(codes=code, save_path=save_path)

        elif self.img_type == 'ascii':
            for id, code, label in tzip(range(len(inputs_code)), inputs_code, labels):
                save_path = os.path.join(folder_path, f"{txtname}_{id}.png")

                save_csv_data.append([save_path, label])
                self.visualize(codes=code, save_path=save_path)

        # 指定CSV文件的路径
        csv_file_path = os.path.join(folder_path, f"{txtname}.csv")

        # 打开CSV文件进行写入，使用newline=''来确保跨平台兼容性
        with open(csv_file_path, mode='w', newline='') as file:
            # 创建CSV写入器
            writer = csv.writer(file)

            # 写入标题行
            writer.writerow(["Path", "Label"])

            # 循环写入数据
            for row in save_csv_data:
                writer.writerow(row)

        print(f"CSV文件已成功写入到 {csv_file_path}")


