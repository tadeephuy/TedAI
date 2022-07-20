from ..imports import *
from ..utils import *

class TedInference:
    """
    Inference class to avoid loading 
    unnecessary stuffs when using Learner class.
    """
    def __init__(self, model, data=None, device=None, loss_func=None, model_path=''):
        self.data, self.model, self.loss_func, self.model_path = data, model, loss_func, model_path
        if self.device is None:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.to(self.device)

    def get_preds(self):
        """import from .utils"""
        pass

    @torch.no_grad()
    def predict(self, x, imread_func=None, post_func=None, **imread_func_kwargs):
        """
        single sample prediction.
        can use as TTA with appropriate `imread_func` and `post_func`

        Args:
            x: input (can be image or path string, depends on the `imread_func`)
            imread_func: function to read and preprocess image
            post_func: function to process prediction
            **imread_func_kwargs: arguments for `imread_func`
        """
        imread_func = self.imread_func if imread_func is None else imread_func
        post_func = self.post_func if post_func is None else post_func

        self.model.eval()
        x = imread_func(x, **imread_func_kwargs).to(self.device)
        x = self.model(x)
        x = post_func(x)
        return x

    def to(self, device):
        self.model = self.model.to(device)
        if self.loss_func is not None:
            self.loss_func = self.loss_func.to(device)

    def load(self, name='model'):
        model_path = os.path.join(self.model_path, f'{name}.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model_state_dict = state_dict.get('model_state_dict', state_dict)
        self.model.load_state_dict(model_state_dict, strict=True)
        print(f'Model is loaded from {model_path}')

    @staticmethod
    def post_func(x): return x[0]

    @staticmethod
    def imread_func(img_path, img_size=256):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_aspect_ratio(img, img_size, interp=cv2.INTER_LINEAR)
        img = min_edge_crop(img, 'center')
        img = np.transpose(img, (2, 0, 1))
        img = img / 255
        imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
        imagenet_std = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])
        img = (img-imagenet_mean)/imagenet_std
        return torch.FloatTensor(img)

from .utils import *