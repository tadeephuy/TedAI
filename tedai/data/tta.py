from ..imports import *
from ..utils import *
from torch.utils.data import ConcatDataset
from sklearn.model_selection import ParameterGrid
from . import TedData

tta_choice = namedtuple('tta_choice', ['weight', 'transform'])

class TestTimeAugmentationData(TedData):
    """
    Databunch for Test Time Augmentation inference.
    """
    def __init__(self, data_path, ds_class, ttas=None, read_resize=None, img_size=224, bs=32, n_workers=8):
        self.data_path, self.ds_class = data_path, ds_class
        self.img_size, self.bs, self.n_workers = img_size, bs, n_workers
        
        self.read_resize = self.default_read_resize if read_resize is None else read_resize
        self.ttas = ttas
        self.tta_chains = self._generate_tta_chains(self.ttas)
        self._initialize_data()
    
    @staticmethod
    def default_read_resize(img_size):
        return [ToPILImage(), Resize(int(img_size*1.05))]
    
    def show_ttas(self):
        display({i:v for i,v in enumerate(list(self.tta_chains))})

    def _generate_tta_chains(self, augs):
        param_grids = {}
        for i, ag in enumerate(augs):
            param_grids[i] = ag
        return ParameterGrid([param_grids])
        

    def _initialize_data(self):
        torch.cuda.empty_cache()
        gc.collect()
        
        self.ds = []
        for tta in self.tta_chains:
            # a function that accepts `img_size` argument
            tta_chain = lambda img_size: Compose(
                self.read_resize(img_size) + [v for _,v in tta.items() if v.__repr__() != 'Identity()']
            )

            self.ds.append(self._create_ds(self.ds_class, 
                                           transforms=tta_chain, 
                                           img_size=self.img_size))
        self.ds = ConcatDataset(self.ds)
        self.dl = self._create_dl(self.ds, shuffle=False, drop_last=False)
    
    def get_tta_dataloader(self, tta_index=0):
        """
        return a dataloader for the 
        tta combination at `tta_index`
        """
        tta_ds = self.ds.datasets[tta_index]
        print('Dataloader with transforms:\n', tta_ds.transforms)
        return self._create_dl(tta_ds, shuffle=False, drop_last=False)

    def show_batch(self, n_row=1, n_col=8, tta_index=None):
        """
        function to show images batch that in 
        each tta configuration
        
        Args:
            n_row: number of images in a row
            n_col: number of images in a column
            tta_index: the index of tta choice in `self.tta_chains`
        """
        n_row = self.bs if n_row >= self.bs else n_row
        
        if tta_index is not None:
            ds = self.ds.datasets[tta_index]
            print('TTA transforms:\n', ds.transforms)
        else:
            ds = self.ds

        for _ in range(n_row):
            idx = np.random.randint(len(ds), size=n_col)
            xb = torch.stack([ds[i][0] for i in idx], dim=0)
            make_imgs(xb, n_row=n_col, plot=True)

    def export_tta_configs(self, name='tta_configs.pkl', tta_configs={},  **kwargs):
        """
        Args:
            - name: filename to save the config
            - tta_configs: a dictionary in form `{tta_idx: weight}`
            - kwargs: other information needed to save
        """

        for k, v in tta_configs.items():
            trans = self.ds.datasets[k].transforms
            tta_configs[k] = tta_choice(v, trans)


        to_save = {'configs': tta_configs, 'others':kwargs, 'ttas': self.ttas, }

        with open(name, 'wb') as f:
            pkl.dump(to_save, f)
            print(f'write tta config file to {name}')

        return to_save

# alias
TTAData = TestTimeAugmentationData