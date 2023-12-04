import os.path as osp
from typing import List, Union
from modelscope.models.base import TorchModel,Model
from modelscope.models.builder import MODELS
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.outputs import OutputKeys
from .background_generation import BackroundGeneration
from PIL import Image

@MODELS.register_module('background_generation_task', module_name='background_generation_model')
class BackgroundGenerationV1Model(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model=BackroundGeneration(model_dir,norm_file=osp.join(model_dir,'cond_image_embeds_mean_std.npy'))
    
    def forward(self,main_img, cond_img, *args, **kwargs):
        return self.model(main_img,cond_img, *args, **kwargs)

@PIPELINES.register_module('background_generation_task','background_generation_v1_pipeline')
class BackgroundGenerationV1Pipeline(Pipeline):
    def __init__(self, config_file: str = None, model = None, **kwargs):
        super().__init__(config_file, model, **kwargs)
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        super().__init__(model=pipe_model,**kwargs)

    def __call__(self, main_img:Union[str,Image.Image], cond_img:Union[str,Image.Image],*args,**kwargs):
        input = (main_img,cond_img)
        return super().__call__(input,*args,**kwargs)
         
    def forward(self, input, **kwargs):
        main_img,cond_img=input
        imgs=self.model(main_img,cond_img, **kwargs)
        out = {
            OutputKeys.OUTPUT_IMGS: imgs
        }
        return out
    
    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output
        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, pipeline_parameters, {}
    
    def postprocess(self, inputs):
        return inputs
    
    def preprocess(self, inputs):
        return inputs
