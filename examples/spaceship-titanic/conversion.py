from typing import Dict, List, Callable
import numpy as np
import torch
import inspect

class Converter:
    _singleton_cvtr = None

    @classmethod
    def covtr(cls):
        if not cls._singleton_cvtr:
            cls._singleton_cvtr = cls()
        return cls._singleton_cvtr

    # Alias por compatibilidad
    @classmethod
    def cvtr(cls):
        return cls.covtr()

    def __init__(self):
        self.order_list: List[str] = []
        self.model_ctor: Callable[..., torch.nn.Module] = None  # puede o no aceptar in_features

    def set_model_ctor(self, ctor: Callable[..., torch.nn.Module]):
        self.model_ctor = ctor

    # ---------- utilidades ----------
    def _infer_in_features_from_models(self, models: Dict[str, np.ndarray]) -> int:
        arr = models.get("fc1_0", None)
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return int(arr.shape[1])
        for _k, a in models.items():
            if isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] > 0:
                return int(a.shape[1])
        raise ValueError("No pude inferir in_features de los pesos recibidos.")

    def _ordered_keys(self, models: Dict[str, np.ndarray]) -> List[str]:
        if self.order_list:  # respeta orden de export previo
            return list(self.order_list)
        def ksort(k: str):
            parts = k.split("_")
            base = parts[0]
            try:
                idx = int(parts[1])
            except Exception:
                idx = 0
            return (base, idx)
        return sorted(models.keys(), key=ksort)

    # ---------- conversiones ----------
    def convert_nn_to_dict_nparray(self, net) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        self.order_list = []  # reset por ronda
        layers = vars(net)['_modules']
        for lname, layer in layers.items():
            for i, ws in enumerate(layer.parameters()):
                mname = f"{lname}_{i}"
                d[mname] = ws.data.detach().cpu().numpy()
                self.order_list.append(mname)
        return d

    def convert_dict_nparray_to_nn(self, models: Dict[str, np.ndarray]):
        assert self.model_ctor is not None, "Model ctor not set in Converter"

        inferred_in = self._infer_in_features_from_models(models)

        # Instanciar respetando firma del ctor (con o sin in_features)
        try:
            sig = inspect.signature(self.model_ctor)
            if len(sig.parameters) >= 1:
                net = self.model_ctor(inferred_in)
            else:
                net = self.model_ctor()
        except TypeError:
            net = self.model_ctor()

        layers = vars(net)['_modules']
        keys = self._ordered_keys(models)
        npa_iter = iter(models[k] for k in keys)
        for _lname, layer in layers.items():
            for ws in layer.parameters():
                arr = next(npa_iter)
                t = torch.from_numpy(arr).to(ws.data.dtype)
                ws.data = t.clone().detach()
        return net

    def get_model_names(self, net) -> List[str]:
        d = self.convert_nn_to_dict_nparray(net)
        return list(d.keys())
