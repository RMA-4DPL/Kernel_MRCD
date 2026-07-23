import numpy as np

class IARR():
    def __init__(self):
        pass

    def transform(self, X):
        """
        Atmospheric correction based on Internal Average Relative Reflectance.
        
        Parameters:
        -----------
        X: ndarray of shape (L, H, W, B)
            An array containing all input hyperspectral image cubes (Height, Width, Bands).
        
        Returns:
        --------
        X: ndarray of shape (L, H, W, B)
            An array containing all hyperspectral image cubes (Height, Width, Bands) with the mean of each band set to 1.
        
        """
        return X/np.mean(X,axis=(1,2), keepdims=True) 

class PassThrough():
    def __init__(self):
        pass

    def transform(self, X):
        return X

def create_AC_model(model_name='IARR'):
    AC_dict = {'IARR': IARR(),
               'none': PassThrough()}

    if model_name in AC_dict:
        return AC_dict[model_name]
    else:
        print(f'{model_name} not known.')

class StandardScaler():
    def __init__(self, scaling_scope='global'):
        self.scaling_scope = scaling_scope
    
    def fit(self, X):
        if self.scaling_scope == 'global':
            axis_to_reduce = tuple(i for i in range(X.ndim) if i != X.ndim-1)
            self.mean = np.zeros((1,1,1,X.shape[-1]), dtype=np.float32)
            self.std = np.zeros((1,1,1,X.shape[-1]), dtype=np.float32)
            for i in range(X.shape[-1]):
                self.mean[:,:,:,i] = np.mean(X[:,:,:,i], keepdims=True)
                self.std[:,:,:,i] = np.std(X[:,:,:,i], keepdims=True)
        elif self.scaling_scope == 'all':
            self.mean = np.mean(X, keepdims=True)
            self.std = np.std(X, keepdims=True)

    def transform(self, X):
        if self.scaling_scope == 'global' or self.scaling_scope=='all':
            return (X - self.mean) / self.std
        else:
            axis_to_reduce = tuple(i for i in range(X.ndim) if i not in (0, X.ndim-1))
            return (X - np.mean(X, axis=axis_to_reduce, keepdims=True)) / np.std(X, axis=axis_to_reduce, keepdims=True)
    
def select_scaler(scaler_name, scaling_scope='global'):
    scaler_dict = {'Standard': StandardScaler(scaling_scope=scaling_scope),
                   'none': PassThrough()}

    if scaler_name in scaler_dict:
        return scaler_dict[scaler_name]
    else:
        print(f"{scaler_name} scaler is not currently supported.")

def subsample_data(X, subsample_factor):
    if subsample_factor is None:
        return X
    else:
        return X[:,::subsample_factor, ::subsample_factor]
    
def scale_vis_bands(X, scale_factor):
    if scale_factor is None:
        return X
    else:
        X[:,:,:,0:3] = X[:,:,:,0:3] / scale_factor
        return X


class PreprocPipeline():
    def __init__(self, experiment_settings):
        self.preproc_pipeline = self.create_preproc_pipeline(experiment_settings)

    def fit(self, X):
        for preproc in self.preproc_pipeline:
            if hasattr(preproc, 'fit'):
                preproc.fit(X)
            if hasattr(preproc, 'transform'):
                X = preproc.transform(X)
            else:
                X = preproc(X)

    def transform(self, X):
        for preproc in self.preproc_pipeline:
            if hasattr(preproc, 'transform'):
                X = preproc.transform(X)
            else:
                X = preproc(X)
        return X

    def create_preproc_pipeline(self, experiment_settings):
        preproc_pipeline = []
        # if 'Subsample' in experiment_settings:
        #     preproc_pipeline.append(lambda X: subsample_data(X, experiment_settings['Subsample']))
        if 'AC_model' in experiment_settings:
            preproc_pipeline.append(create_AC_model(experiment_settings['AC_model']['name']))
        if 'Scaler' in experiment_settings:
            preproc_pipeline.append(select_scaler(experiment_settings['Scaler']['name'],
                                                    experiment_settings['Scaler'].get('scaling_scope', 'global')))
        if 'Vis_scale' in experiment_settings:
            preproc_pipeline.append(lambda X: scale_vis_bands(X, experiment_settings['Vis_scale']))

        return preproc_pipeline