dimport os
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

base_skin_dir = os.path.join('backend', 'data') #base_skin_dir = data
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
#imageid_path_dict = {'ISIC_0024306': 'data/HAM10000_images_part_1/ISIC_0024306.jpg'}

#对7种皮肤癌种类进行标号
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))# 存放标签数据

# path对应路径，cell_type对应详情，cell_type_idx对应病症
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get) # 获取图像路径
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) # 病变的详情
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

# 以9：0.5：0.5的比例划分数据集
train_df, test_df = train_test_split(tile_df, test_size=0.1)
validation_df, test_df = train_test_split(test_df, test_size=0.5)

# 索引重置
train_df = train_df.reset_index()
validation_df = validation_df.reset_index()
test_df = test_df.reset_index()

# 数据集
class SkinDataset(data.Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    # 返回图片和对应的标签
    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        # 数据增强
        if self.transform:
            X = self.transform(X)

        return X, y
