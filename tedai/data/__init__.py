from ..imports import *
from ..utils import *

class TedData:
    """
    Fastai databunch immitation
    """
    def __init__(self, data_path, ds_class, transforms, img_size=224, bs=32, n_workers=8, fix_dis=False):
        """
        Args:
            data_path: str - path to the data
            ds_class: tuple | class - 
                      (train_dataset_class, val_dataset_class) or 
                      (train_dataset_class, val_dataset_class, test_dataset_class)
                      can pass train_dataset_class ->  (train_dataset_class, train_dataset_class) 
            transforms: tuple | class - (train_transforms, val_transforms) can pass train_transforms -> (train_transforms, None)
        """
        self.data_path = data_path
        self.img_size, self.bs, self.fix_dis = img_size, bs, fix_dis
        self.n_workers = n_workers

        if not isinstance(ds_class, tuple): 
            self.ds_class = (ds_class, ds_class)
        else: self.ds_class = ds_class

        if isinstance(transforms, tuple): 
            self.transforms = transforms
        else: self.transforms = (transforms, None)

        self._initialize_data()

    def set_size(self, img_size, bs=None, n_workers=None):
        """
        function to set img size for the Data
        """
        self.img_size = img_size
        self.bs = bs or self.bs
        self.n_workers = n_workers or self.n_workers
        self._initialize_data()

    def _initialize_data(self):
        torch.cuda.empty_cache()
        gc.collect()

        self.train_ds = self._create_ds(self.ds_class[0], transforms=self.transforms[0], img_size=self.img_size)
        self.val_ds = self._create_ds(self.ds_class[1], transforms=self.transforms[1], img_size=self.img_size, fix_dis=self.fix_dis)
        self.train_dl = self._create_dl(self.train_ds, shuffle=True)
        self.val_dl = self._create_dl(self.val_ds, shuffle=False)
        self.test_ds = None
        if len(self.ds_class) == 3:
            self.add_test(test_ds=self.ds_class[2])
        else:
            self.test_dl = None

    def _create_ds(self, ds_class, transforms=None, img_size=None, fix_dis=False, **kwargs):
        if fix_dis: 
            img_size = int(1.4*img_size)
        return ds_class(data_path=self.data_path, transforms=transforms, img_size=img_size, **kwargs)

    def _create_dl(self, dataset, shuffle, bs=None, **kwargs):
        return DataLoader(dataset=dataset, batch_size=bs or self.bs, shuffle=shuffle, 
                          num_workers=self.n_workers, pin_memory=True, **kwargs)
    
    def add_test(self, test_ds):
        self.test_ds = self._create_ds(test_ds, transforms=self.transforms[1], img_size=self.img_size, fix_dis=self.fix_dis)
        self.test_dl = self._create_dl(self.test_ds, shuffle=False)

    def show_batch(self, n_row=1, n_col=8, mode='train'):
        """
        function to show images batch that is fed for training
        
        Args:
            n_row: number of images in a row
            n_col: number of images in a column
            mode: `train` or `valid` to specify the img from each session
        """
        n_row = self.bs if n_row >= self.bs else n_row
        ds = {
            'train': self.train_ds, 
            'valid': self.val_ds, 
            'test': self.test_ds or self.val_ds
        }.get(mode, self.train_ds)

        for _ in range(n_row):
            idx = np.random.randint(len(ds), size=n_col)
            xb = torch.stack([ds[i][0] for i in idx], dim=0)
            make_imgs(xb, n_row=n_col, plot=True)

class TedImageDataset(Dataset):
    def __init__(self, data_path, df, transforms=None, img_size=224, label_cols_list=None):
        self.data_path,self.img_size,self.df = data_path,img_size,df
        self.transforms = transforms or self.default_transforms()
        self.transforms = partial(self.transforms, img_size=self.img_size)()
        
        # labels
        self.label_cols_list = label_cols_list
        if self.label_cols_list is not None:
            self.labels = self.df[self.label_cols_list].to_numpy()
        
    def __getitem__(self, idx):
        img = self._imread(self.df.Images[idx])
        if self.label_cols_list is not None:
            label = self.labels[idx]
            return self.transforms(img), label.astype(float)
        return self.transforms(img)

    def __len__(self): return len(self.df)

    def _imread(self, img_path): 
        return cv2.cvtColor(cv2.imread(os.path.join(self.data_path, img_path)), cv2.COLOR_BGR2RGB)

    @staticmethod
    def default_transforms():
        return lambda img_size: Compose([ToPILImage(), Resize(int(img_size*1.3)), CenterCrop((img_size, img_size)), ToTensor(), 
                                               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])