import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from crp.cache import Cache
from crp.concepts import Concept
from typing import Dict,List, Tuple
from crp.statistics import Statistics
from zennit.composites import Composite
from crp.maximization import Maximization
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization


class ConceptMaximization(Maximization):
    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):
        super().__init__(mode, max_target, abs_norm, path)

    def _save_results(self, d_index: Tuple[int, int] = None):
        """same as max"""
        saved_files = []

        for layer_name in self.d_c_sorted:
            if d_index:
                filename = f"{layer_name}_{d_index[0]}_{d_index[1]}_"
            else:
                filename = f"{layer_name}_"

            if '\\' in filename:
                filename = filename.rsplit('\\', 1)[-1]

            np.save(self.PATH / Path(filename + "data.npy"), self.d_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rf.npy"), self.rf_c_sorted[layer_name].cpu().numpy())
            np.save(self.PATH / Path(filename + "rel.npy"), self.rel_c_sorted[layer_name].cpu().numpy())

            saved_files.append(str(self.PATH / Path(filename)))

        self.delete_result_arrays()
        return saved_files



class ConceptStatistics(Statistics):
    def __init__(self, mode="relevance", max_target="sum", abs_norm=False, path=None):
        super().__init__(mode, max_target, abs_norm, path)

    def collect_results(self, path_list: List[str], d_index: Tuple[int, int] = None):
        """same as stat"""
        self.delete_result_arrays()

        pbar = tqdm(total=len(path_list), dynamic_ncols=True)

        for path in path_list:

            l_name, filename = path.split("/")[-2:]
            target = filename.split("_")[0]

            d_c_sorted = np.load(path + "data.npy")
            rf_c_sorted = np.load(path + "rf.npy")
            rel_c_sorted = np.load(path + "rel.npy")

            d_c_sorted, rf_c_sorted, rel_c_sorted = map(torch.from_numpy, [d_c_sorted, rf_c_sorted, rel_c_sorted])

            self.concatenate_with_results(l_name, target, d_c_sorted, rel_c_sorted, rf_c_sorted)
            self.sort_result_array(l_name, target)

            pbar.update(1)

        for path in path_list:
            for suffix in ["data.npy", "rf.npy", "rel.npy"]:
                os.remove(path + suffix)

        pbar.close()

        return self._save_results(d_index)



class ConceptRelevanceAttribute(CondAttribution):
    def __init__(self, model: nn.Module, device=None,overwrite_data_grad=True, no_param_grad=True) -> None:
        if device is None:
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model, device,overwrite_data_grad, no_param_grad)

    def relevance_init(self, prediction, target_list, init_rel):
        if callable(init_rel):
            output_selection = init_rel(prediction)
        elif isinstance(init_rel, torch.Tensor):
            output_selection = init_rel
        elif isinstance(init_rel, (int, np.integer)):
            output_selection = torch.full(prediction.shape, init_rel)
        else:
            output_selection = prediction
       
        if target_list:
            mask = torch.zeros_like(output_selection)
            for i, targets in enumerate(target_list):
                mask[i, targets] = output_selection[i, targets]
            output_selection = mask
        #print(prediction, target_list,output_selection)
        return output_selection



class ConceptVisualization(FeatureVisualization):
    def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept], preprocess_fn = None, max_target="sum", abs_norm=True, path="ConceptVisualization", device=None, cache: Cache = None):
        super().__init__(attribution, dataset, layer_map, preprocess_fn, max_target, abs_norm, path, device, cache)
        self.RelMax = ConceptMaximization("relevance", max_target, abs_norm, path)
        self.ActMax = ConceptMaximization("activation", max_target, abs_norm, path)

        self.RelStats = Statistics("relevance", max_target, abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

    def run(self, composite: Composite, data_start=0, data_end=1000, batch_size=16, checkpoint=250, on_device=None):
        print("Running Analysis...")
        saved_checkpoints = self.run_distributed(composite, data_start, data_end, batch_size, checkpoint, on_device)

        print("Collecting concepts...")
        saved_files = self.collect_results(saved_checkpoints)
        return saved_files
    
    
